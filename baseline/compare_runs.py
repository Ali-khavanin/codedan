"""Compare an experiment run against the frozen baseline.

Both directories use the same layout: <run_dir>/<benchmark>/summary.json.

    python baseline/compare_runs.py baseline/results/qwen3-8b-base-fp8_v1 runs/<experiment>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARKS = ["mbpp", "mbpp_plus", "livecodebench_v5"]


def load(run_dir: Path, bench: str) -> dict:
    s = json.loads((run_dir / bench / "summary.json").read_text())
    m = s["metrics"]
    k = s["samples_per_task"]
    return {
        "accuracy": m[f"avg_at_{k}"],
        "reasoning_tokens": m["mean_reasoning_tokens"],
        "latency_sec": m["mean_latency_sec"],
        "wall_clock_sec": m["total_wall_clock_sec"],
    }


def compare(baseline_dir: Path, run_dir: Path) -> list[dict]:
    rows = []
    for bench in BENCHMARKS:
        base, run = load(baseline_dir, bench), load(run_dir, bench)
        rows.append(
            {
                "benchmark": bench,
                "baseline_accuracy": base["accuracy"],
                "run_accuracy": run["accuracy"],
                "delta_accuracy_pp": round(100 * (run["accuracy"] - base["accuracy"]), 2),
                "delta_reasoning_tokens": round(run["reasoning_tokens"] - base["reasoning_tokens"], 1),
                "delta_latency_sec": (
                    round(run["latency_sec"] - base["latency_sec"], 3)
                    if run["latency_sec"] is not None and base["latency_sec"] is not None
                    else None
                ),
                "delta_wall_clock_sec": round(run["wall_clock_sec"] - base["wall_clock_sec"], 1),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    missing = [str(d / b / "summary.json") for d in (args.baseline_dir, args.run_dir) for b in BENCHMARKS
               if not (d / b / "summary.json").is_file()]
    if missing:
        print("missing summaries:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    rows = compare(args.baseline_dir, args.run_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    cols = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
