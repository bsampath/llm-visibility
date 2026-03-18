#!/usr/bin/env python3
"""
Benchmark harness for vLLM's OpenAI-compatible /v1/completions endpoint.

Sends N requests at a given concurrency level and emits one JSON-lines record
per completed request to stdout, followed by a summary to stderr.

Usage:
    python run_inference.py --model meta-llama/Llama-3-8B-Instruct \
        --concurrency 8 --num-requests 100 \
        --prompt-tokens 512 --max-tokens 256 \
        > results/run_c8_p512_g256.jsonl

Dry-run against the mock server:
    python run_inference.py --host localhost --port 9999 \
        --model mock --num-requests 20 --concurrency 4 --dry-run
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict

import aiohttp


# ---------------------------------------------------------------------------
# Token-length helpers
# ---------------------------------------------------------------------------

# Rough token-to-word ratio for English (~1.3 tokens/word).
# Used to build a prompt of approximately the requested token length without
# requiring a tokenizer at benchmark time.
_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "how much wood would a woodchuck chuck "
    "she sells seashells by the seashore "
).split()


def _build_prompt(target_tokens: int) -> str:
    """Return a repeating-words prompt that is ~target_tokens tokens long."""
    words_needed = max(1, int(target_tokens / 1.3))
    cycle = (_WORDS * ((words_needed // len(_WORDS)) + 1))[:words_needed]
    return " ".join(cycle)


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    ts: float           # Unix timestamp at request completion
    latency_s: float    # Wall-clock seconds from send to last byte
    ttft_s: float       # Time-to-first-token (–1 if not available)
    tokens_generated: int
    prompt_tokens: int
    max_tokens: int
    concurrency: int
    status: int         # HTTP status code; 0 on connection error
    error: str          # Empty string on success


# ---------------------------------------------------------------------------
# Single async request
# ---------------------------------------------------------------------------

async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> RequestResult:
    prompt_tokens = max(1, int(len(prompt.split()) * 1.3))
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }

    t0 = time.perf_counter()
    ttft = -1.0
    status = 0
    error = ""
    tokens_generated = 0

    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            body = await resp.json(content_type=None)
            t1 = time.perf_counter()

            if status == 200:
                choice = body.get("choices", [{}])[0]
                usage = body.get("usage", {})
                tokens_generated = usage.get("completion_tokens", max_tokens)
                # vLLM echoes first-token timing in some builds; fall back to –1
                ttft = body.get("first_token_time", -1.0)
            else:
                error = body.get("detail", f"HTTP {status}")
    except aiohttp.ClientError as exc:
        t1 = time.perf_counter()
        error = str(exc)

    return RequestResult(
        ts=time.time(),
        latency_s=t1 - t0,
        ttft_s=ttft,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        concurrency=concurrency,
        status=status,
        error=error,
    )


# ---------------------------------------------------------------------------
# Concurrent load driver
# ---------------------------------------------------------------------------

async def run_load(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_requests: int,
    concurrency: int,
) -> list[RequestResult]:
    """
    Fire num_requests requests with at most `concurrency` in flight at once.
    Uses a semaphore so we never exceed the target concurrency.
    """
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    results: list[RequestResult] = []

    async def bounded_request() -> RequestResult:
        async with semaphore:
            return await _send_request(
                session, url, model, prompt, max_tokens, concurrency
            )

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(bounded_request()) for _ in range(num_requests)]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            # Emit the record immediately so callers can stream/pipe it
            print(json.dumps(asdict(result)), flush=True)

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, int(len(sorted_values) * pct / 100) - 1)
    return sorted_values[idx]


def print_summary(results: list[RequestResult], elapsed_wall: float) -> None:
    successes = [r for r in results if r.status == 200]
    errors = [r for r in results if r.status != 200]

    latencies_ms = sorted(r.latency_s * 1000 for r in successes)
    total_tokens = sum(r.tokens_generated for r in successes)

    tokens_per_sec = total_tokens / elapsed_wall if elapsed_wall > 0 else 0
    req_per_sec = len(successes) / elapsed_wall if elapsed_wall > 0 else 0

    lines = [
        "",
        "=" * 60,
        "BENCHMARK SUMMARY",
        "=" * 60,
        f"  Total requests  : {len(results)}",
        f"  Successful      : {len(successes)}",
        f"  Errors          : {len(errors)}",
        f"  Wall time       : {elapsed_wall:.2f}s",
        f"  Throughput      : {req_per_sec:.2f} req/s",
        f"  Tokens/sec      : {tokens_per_sec:.1f}",
        "",
        "  Latency (ms)",
        f"    p50  : {_percentile(latencies_ms, 50):.1f}",
        f"    p95  : {_percentile(latencies_ms, 95):.1f}",
        f"    p99  : {_percentile(latencies_ms, 99):.1f}",
        f"    mean : {sum(latencies_ms)/len(latencies_ms):.1f}" if latencies_ms else "    mean : —",
    ]

    if errors:
        lines += ["", "  Error breakdown:"]
        from collections import Counter
        for msg, count in Counter(r.error for r in errors).most_common():
            lines.append(f"    [{count}x] {msg}")

    lines.append("=" * 60)
    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Configurable vLLM inference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="meta-llama/Llama-3-8B-Instruct")
    p.add_argument("--num-requests", type=int, default=100,
                   help="Total number of requests to send")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max requests in flight simultaneously")
    p.add_argument("--prompt-tokens", type=int, default=128,
                   help="Approximate input prompt length in tokens")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Number of tokens to generate per request")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the actual run; just print the config and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/completions"
    prompt = _build_prompt(args.prompt_tokens)

    config = {
        "url": url,
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
    }
    print(f"# config {json.dumps(config)}", file=sys.stderr)

    if args.dry_run:
        print("Dry-run: exiting without sending requests.", file=sys.stderr)
        return

    t_start = time.perf_counter()
    results = asyncio.run(
        run_load(
            url=url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.perf_counter() - t_start
    print_summary(results, elapsed)


if __name__ == "__main__":
    main()
