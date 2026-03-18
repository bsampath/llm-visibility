#!/usr/bin/env python3
"""
Minimal mock server that mimics vLLM's OpenAI-compatible /v1/completions endpoint.

Designed for testing benchmark/run_inference.py locally without a GPU or a real
model. It validates the request shape, sleeps to simulate generation latency, and
returns a response in the same format vLLM uses.

Usage:
    python benchmark/mock_server.py --port 9999

    # Then in another terminal:
    python benchmark/run_inference.py \
        --host localhost --port 9999 --model mock \
        --num-requests 20 --concurrency 4

Flags:
    --port          Port to listen on (default: 9999)
    --latency-ms    Base latency per request in ms (default: 200)
    --latency-per-token-ms  Extra ms added per generated token (default: 5)
    --error-rate    Fraction of requests to fail with HTTP 500 (default: 0.0)
"""

import argparse
import asyncio
import json
import random
import time
import uuid

from aiohttp import web


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

async def completions(request: web.Request) -> web.Response:
    args: argparse.Namespace = request.app["args"]

    # Parse body
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="Bad JSON")

    # Validate required fields
    for field in ("model", "prompt", "max_tokens"):
        if field not in body:
            return web.Response(
                status=422,
                content_type="application/json",
                text=json.dumps({"detail": f"Missing field: {field}"}),
            )

    max_tokens: int = int(body["max_tokens"])
    prompt: str = body["prompt"]
    prompt_tokens = max(1, int(len(prompt.split()) * 1.3))

    # Simulate random errors
    if random.random() < args.error_rate:
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"detail": "mock server injected error"}),
        )

    # Simulate generation latency
    delay_s = (args.latency_ms + args.latency_per_token_ms * max_tokens) / 1000.0
    await asyncio.sleep(delay_s)

    # Build a vLLM-shaped response
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    generated_text = " ".join(["token"] * max_tokens)

    response_body = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        },
    }

    return web.Response(
        status=200,
        content_type="application/json",
        text=json.dumps(response_body),
    )


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application()
    app["args"] = args
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/health", health)
    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mock vLLM server for local benchmark testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--latency-ms", type=int, default=200,
                   help="Base latency per request (ms)")
    p.add_argument("--latency-per-token-ms", type=int, default=5,
                   help="Additional latency per generated token (ms)")
    p.add_argument("--error-rate", type=float, default=0.0,
                   help="Fraction of requests to return HTTP 500 (0.0–1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)

    base_latency = args.latency_ms
    per_token = args.latency_per_token_ms
    print(
        f"Mock vLLM server listening on http://0.0.0.0:{args.port}\n"
        f"  Base latency         : {base_latency} ms\n"
        f"  Per-token latency    : {per_token} ms\n"
        f"  Error rate           : {args.error_rate:.0%}\n"
        f"  Endpoints            : POST /v1/completions  GET /health"
    )
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
