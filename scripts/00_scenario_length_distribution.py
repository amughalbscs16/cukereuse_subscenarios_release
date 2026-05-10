"""Phase 0 — scenario-length distribution sanity check.

Groups the 1.1M-row step table by (repo, scenario) and reports the
distribution of scenario lengths. Used to ground the L_max parameter
for the slice-extraction phase: we'll cap L at roughly p99 of this
distribution.

Outputs:
- analysis/scenario_length_distribution.json (summary statistics)
- analysis/scenario_length_histogram.csv     (length -> count)
- console table

Run with the project venv from the repo root:
    python scripts/00_scenario_length_distribution.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from _parquet_loader import load_steps

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def main() -> None:
    steps = load_steps(columns=["repo", "scenario"])
    print(f"loaded steps: {len(steps):,} rows")

    scenario_lengths = steps.groupby(["repo", "scenario"]).size()
    print(f"scenarios: {len(scenario_lengths):,}")

    arr = scenario_lengths.to_numpy()
    summary = {
        "n_steps": int(len(steps)),
        "n_scenarios": int(len(scenario_lengths)),
        "n_repos": int(steps["repo"].nunique()),
        "min": int(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
        "max": int(arr.max()),
        "n_length_lt_2": int((arr < 2).sum()),
        "n_length_eq_1": int((arr == 1).sum()),
        "n_length_eq_0": int((arr == 0).sum()),
    }

    print("\n=== scenario length distribution ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>15}: {v:>10.2f}")
        else:
            print(f"  {k:>15}: {v:>10,}")

    out_json = ANALYSIS_DIR / "scenario_length_distribution.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")

    histogram = scenario_lengths.value_counts().sort_index()
    out_csv = ANALYSIS_DIR / "scenario_length_histogram.csv"
    histogram.to_csv(out_csv, header=["count"], index_label="length")
    print(f"wrote {out_csv} ({len(histogram)} distinct lengths)")

    print("\n=== suggested L_max ===")
    p99_int = int(np.ceil(summary["p99"]))
    p95_int = int(np.ceil(summary["p95"]))
    print(f"  if L_max = p95: {p95_int}")
    print(f"  if L_max = p99: {p99_int}")
    print(f"  if L_max = max: {summary['max']} (likely too large; outliers)")


if __name__ == "__main__":
    main()
