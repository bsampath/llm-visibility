#!/usr/bin/env python3
"""
benchmark/collect_baseline.py

Drives run_inference.py across all benchmark combinations and saves raw
JSON-lines output for each run. Calls results_to_csv.py at the end to
produce a single summary CSV.

Usage:
    # Against a real vLLM server:
    python benchmark/collect_baseline.py \
        --model meta-llama/Llama-3-8B-Instruct --port 8000

    # Against the mock server (auto-started):
    python benchmark/collect_baseline.py --mock

    # Customise the sweep:
    python benchmark/collect_baseline.py --mock \
        --concurrency 1 4 8 --prompt-tokens 128 512 --max-tokens 64 \
        --num-requests 50
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo root is one level above this file
REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
BENCHMARK_DIR = REPO_ROOT / "benchmark"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Mock server lifecycle
# ---------------------------------------------------------------------------

async def start_mock_server(port: int) -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        str(VENV_PYTHON),
        str(BENCHMARK_DIR / "mock_server.py"),
        "--port", str(port),
        "--latency-ms", "100",
        "--latency-per-token-ms", "2",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(1)  # give the server a moment to bind
    log(f"Mock server started (PID {proc.pid}) on port {port}")
    return proc


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

async def run_once(
    *,
    host: str,
    port: int,
    model: str,
    num_requests: int,
    concurrency: int,
    prompt_tokens: int,
    max_tokens: int,
    outfile: Path,
    logfile: Path,
) -> bool:
    """
    Invokes run_inference.py as a subprocess, writing JSON-lines to outfile
    and stderr to logfile. Returns True on success.
    """
    cmd = [
        str(VENV_PYTHON),
        str(BENCHMARK_DIR / "run_inference.py"),
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--num-requests", str(num_requests),
        "--concurrency", str(concurrency),
        "--prompt-tokens", str(prompt_tokens),
        "--max-tokens", str(max_tokens),
    ]

    with outfile.open("w") as stdout_f, logfile.open("a") as stderr_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        await proc.wait()

    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def sweep(args: argparse.Namespace) -> None:
    results_dir = REPO_ROOT / args.results_dir
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    mock_proc = None
    if args.mock:
        args.port = 9999
        args.model = "mock"
        mock_proc = await start_mock_server(args.port)

    try:
        await _run_sweep(args, results_dir, raw_dir)
    finally:
        if mock_proc is not None:
            try:
                mock_proc.terminate()
                await mock_proc.wait()
            except ProcessLookupError:
                pass  # already exited
            log("Mock server stopped.")


async def _run_sweep(
    args: argparse.Namespace,
    results_dir: Path,
    raw_dir: Path,
) -> None:
    combinations = [
        (c, p, g)
        for c in args.concurrency
        for p in args.prompt_tokens
        for g in args.max_tokens
    ]
    total = len(combinations)

    # Write run metadata
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency_levels": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "total_runs": total,
    }
    meta_path = results_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"Metadata written to {meta_path.relative_to(REPO_ROOT)}")
    log(f"Starting sweep: {total} runs × {args.num_requests} requests each\n")

    log_path = results_dir / "benchmark.log"
    failed: list[str] = []

    for idx, (concurrency, prompt_tokens, max_tokens) in enumerate(combinations, 1):
        tag = f"c{concurrency}_p{prompt_tokens}_g{max_tokens}"
        outfile = raw_dir / f"{tag}.jsonl"

        log(f"[{idx}/{total}] concurrency={concurrency}  prompt_tokens={prompt_tokens}  max_tokens={max_tokens}")

        t0 = time.perf_counter()
        ok = await run_once(
            host=args.host,
            port=args.port,
            model=args.model,
            num_requests=args.num_requests,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            outfile=outfile,
            logfile=log_path,
        )
        elapsed = time.perf_counter() - t0

        if ok:
            lines = sum(1 for _ in outfile.open())
            log(f"  -> {lines} records saved to {outfile.name}  ({elapsed:.1f}s)\n")
        else:
            log(f"  -> FAILED (see {log_path.name})\n")
            failed.append(tag)

    log(f"Sweep complete: {total} runs, {len(failed)} failed.")
    if failed:
        log(f"Failed: {', '.join(failed)}")

    # Parse raw output → CSV
    log("Parsing results to CSV...")
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(BENCHMARK_DIR / "results_to_csv.py"),
            "--raw-dir", str(raw_dir),
            "--output", str(results_dir / "summary.csv"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log(f"Summary CSV written to {args.results_dir}/summary.csv")
    else:
        log(f"results_to_csv.py failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect baseline benchmark results across all configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="meta-llama/Llama-3-8B-Instruct")
    p.add_argument("--num-requests", type=int, default=200,
                   help="Requests per run")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16],
                   help="Concurrency levels to sweep")
    p.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 512, 1024],
                   help="Prompt lengths (tokens) to sweep")
    p.add_argument("--max-tokens", type=int, nargs="+", default=[64, 256],
                   help="Generation lengths (tokens) to sweep")
    p.add_argument("--results-dir", default="results/week1_baseline",
                   help="Output directory relative to repo root")
    p.add_argument("--mock", action="store_true",
                   help="Auto-start the mock server (sets host=localhost, port=9999, model=mock)")
    return p.parse_args()


if __name__ == "__main__":
    # Validate venv exists before doing anything
    if not VENV_PYTHON.exists():
        sys.exit(
            f"ERROR: venv not found at {VENV_PYTHON}\n"
            "Run: python3 -m venv .venv && .venv/bin/pip install aiohttp"
        )

    asyncio.run(sweep(parse_args()))
