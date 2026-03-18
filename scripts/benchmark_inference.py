#!/usr/bin/env python3
"""Placeholder: inference throughput benchmark against a vLLM endpoint."""

import argparse
import time
import requests


def run_benchmark(host: str, port: int, model: str, num_requests: int, prompt: str) -> None:
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 128,
    }

    latencies = []
    for i in range(num_requests):
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=60)
        latencies.append(time.perf_counter() - start)
        resp.raise_for_status()
        print(f"[{i+1}/{num_requests}] latency={latencies[-1]:.3f}s")

    print(f"\nMean latency : {sum(latencies)/len(latencies):.3f}s")
    print(f"Throughput   : {num_requests/sum(latencies):.2f} req/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM inference benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--prompt", default="The quick brown fox")
    args = parser.parse_args()

    run_benchmark(args.host, args.port, args.model, args.num_requests, args.prompt)
