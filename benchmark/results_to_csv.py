#!/usr/bin/env python3
"""
benchmark/results_to_csv.py

Reads all JSON-lines files produced by run_inference.py from a raw/ directory
and writes a single summary CSV — one row per run.

Usage:
    python benchmark/results_to_csv.py \
        --raw-dir results/week1_baseline/raw \
        --output  results/week1_baseline/summary.csv

    # Print to stdout instead:
    python benchmark/results_to_csv.py --raw-dir results/week1_baseline/raw
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "run_tag",
    "concurrency",
    "prompt_tokens",
    "max_tokens",
    "total_requests",
    "successful",
    "errors",
    "wall_time_s",
    "tokens_per_sec",
    "req_per_sec",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = max(0, int(len(sorted_values) * pct / 100) - 1)
    return round(sorted_values[idx], 3)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else float("nan")


def _fmt(value: float) -> str:
    """Format a float for CSV; emit empty string for nan."""
    if value != value:  # nan check
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Parse one JSONL file → one summary row
# ---------------------------------------------------------------------------

def parse_run(path: Path) -> dict:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No valid records in {path}")

    successes = [r for r in records if r.get("status") == 200]
    errors    = [r for r in records if r.get("status") != 200]

    # Pull config from the first record (all records in a file share these)
    first = records[0]
    concurrency  = first.get("concurrency", 0)
    prompt_tokens = first.get("prompt_tokens", 0)
    max_tokens   = first.get("max_tokens", 0)

    # Latency stats (ms) — successes only
    latencies_ms = sorted(r["latency_s"] * 1000 for r in successes)

    # Time-to-first-token stats — only when available (ttft_s != -1)
    ttft_ms = sorted(
        r["ttft_s"] * 1000
        for r in successes
        if r.get("ttft_s", -1) >= 0
    )

    # Throughput: total tokens generated / elapsed wall time
    if successes:
        total_tokens = sum(r["tokens_generated"] for r in successes)
        # Wall time = span from first request sent to last completed
        t_min = min(r["ts"] - r["latency_s"] for r in successes)
        t_max = max(r["ts"] for r in successes)
        wall_time_s = round(t_max - t_min, 3)
        tokens_per_sec = round(total_tokens / wall_time_s, 2) if wall_time_s > 0 else float("nan")
        req_per_sec    = round(len(successes) / wall_time_s, 2) if wall_time_s > 0 else float("nan")
    else:
        wall_time_s    = float("nan")
        tokens_per_sec = float("nan")
        req_per_sec    = float("nan")

    return {
        "run_tag":         path.stem,
        "concurrency":     concurrency,
        "prompt_tokens":   prompt_tokens,
        "max_tokens":      max_tokens,
        "total_requests":  len(records),
        "successful":      len(successes),
        "errors":          len(errors),
        "wall_time_s":     _fmt(wall_time_s),
        "tokens_per_sec":  _fmt(tokens_per_sec),
        "req_per_sec":     _fmt(req_per_sec),
        "latency_mean_ms": _fmt(_mean(latencies_ms)),
        "latency_p50_ms":  _fmt(_percentile(latencies_ms, 50)),
        "latency_p95_ms":  _fmt(_percentile(latencies_ms, 95)),
        "latency_p99_ms":  _fmt(_percentile(latencies_ms, 99)),
        "ttft_mean_ms":    _fmt(_mean(ttft_ms)),
        "ttft_p50_ms":     _fmt(_percentile(ttft_ms, 50)),
        "ttft_p95_ms":     _fmt(_percentile(ttft_ms, 95)),
        "ttft_p99_ms":     _fmt(_percentile(ttft_ms, 99)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        sys.exit(f"ERROR: --raw-dir {raw_dir} does not exist or is not a directory")

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"ERROR: no .jsonl files found in {raw_dir}")

    rows = []
    for path in jsonl_files:
        try:
            rows.append(parse_run(path))
        except Exception as exc:
            print(f"WARNING: skipping {path.name}: {exc}", file=sys.stderr)

    if not rows:
        sys.exit("ERROR: no rows produced — check your raw files")

    # Sort by (concurrency, prompt_tokens, max_tokens) for readability
    rows.sort(key=lambda r: (r["concurrency"], r["prompt_tokens"], r["max_tokens"]))

    # Write
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            _write_csv(f, rows)
        print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    else:
        _write_csv(sys.stdout, rows)


def _write_csv(sink, rows: list[dict]) -> None:
    writer = csv.DictWriter(sink, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse raw benchmark JSON-lines into a summary CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", required=True,
                   help="Directory containing .jsonl files from run_inference.py")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Omit to print to stdout.")
    return p.parse_args()


if __name__ == "__main__":
    main()
