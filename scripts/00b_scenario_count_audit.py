"""Audit the scenario count discrepancy and lock down a defensible
scenario identity for paper 2.

The naive Phase-0 groupby on (repo, scenario) reported 120,532 scenarios
because:
  - 65,278 step rows have empty scenario names (Karate-style `*` steps
    and Background blocks).
  - All those rows for one repo collapse into a single empty-named
    "scenario", inflating the longest group to 8,765 steps (folio-org).

This script tests progressively cleaner scenario keys and reports each
candidate count + length distribution. The cleanest key wins for Phase 1.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from _parquet_loader import load_steps

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def summarise(arr: np.ndarray, label: str) -> dict[str, float]:
    return {
        "label": label,
        "n_scenarios": int(len(arr)),
        "min": int(arr.min()) if len(arr) else 0,
        "p25": float(np.percentile(arr, 25)) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "p75": float(np.percentile(arr, 75)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "max": int(arr.max()) if len(arr) else 0,
        "n_length_lt_2": int((arr < 2).sum()),
    }


def fmt(d: dict[str, float]) -> str:
    return (f"  n={d['n_scenarios']:>8,d}  median={d['median']:>5.1f}  "
            f"p90={d['p90']:>5.1f}  p95={d['p95']:>5.1f}  p99={d['p99']:>6.1f}  "
            f"max={d['max']:>6,d}  len<2={d['n_length_lt_2']:>5,d}")


steps = load_steps()
print("schema:", list(steps.columns))
print(f"rows: {len(steps):,}")

n_bg = int(steps["is_background"].sum())
n_karate_star = int(((steps["keyword"].astype(str).str.strip() == "*") &
                     (steps["scenario"].astype(str).str.strip() == "")).sum())
n_empty_scenario = int((steps["scenario"].astype(str).str.strip() == "").sum())
print(f"\nis_background = True:        {n_bg:>8,d}")
print(f"empty scenario name (any):   {n_empty_scenario:>8,d}")
print(f"  of which `*` (Karate):     {n_karate_star:>8,d}")
print(f"is_outline = True:           {int(steps['is_outline'].sum()):>8,d}")

results = {}

# Key A — naive (repo, scenario), no filtering
arrA = steps.groupby(["repo", "scenario"]).size().to_numpy()
results["A_naive_repo_scenario"] = summarise(arrA, "naive (repo, scenario)")
print("\nA. naive (repo, scenario):")
print(fmt(results["A_naive_repo_scenario"]))

# Key B — (repo_slug, file_path, scenario)
arrB = steps.groupby(["repo_slug", "file_path", "scenario"]).size().to_numpy()
results["B_repo_file_scenario"] = summarise(arrB, "(repo_slug, file_path, scenario)")
print("\nB. (repo_slug, file_path, scenario):")
print(fmt(results["B_repo_file_scenario"]))

# Key C — drop is_background, then (repo_slug, file_path, scenario)
non_bg = steps[~steps["is_background"]]
arrC = non_bg.groupby(["repo_slug", "file_path", "scenario"]).size().to_numpy()
results["C_no_bg_repo_file_scenario"] = summarise(arrC, "no-Background, (repo_slug, file_path, scenario)")
print("\nC. drop is_background, then (repo_slug, file_path, scenario):")
print(fmt(results["C_no_bg_repo_file_scenario"]))

# Key D — also drop empty scenario names (Karate `*` and unnamed)
named = non_bg[non_bg["scenario"].astype(str).str.strip() != ""]
arrD = named.groupby(["repo_slug", "file_path", "scenario"]).size().to_numpy()
results["D_named_no_bg"] = summarise(arrD, "named, no-Background, (repo_slug, file_path, scenario)")
print("\nD. drop is_background AND empty scenario names:")
print(fmt(results["D_named_no_bg"]))

# Key E — D plus drop length < 2 (degenerate one-step scenarios — likely
# pickled outline rows or parser glitches)
arrE = arrD[arrD >= 2]
results["E_named_no_bg_len_ge_2"] = summarise(arrE, "D + length>=2")
print("\nE. D, then length >= 2:")
print(fmt(results["E_named_no_bg_len_ge_2"]))

# Probe the longest scenario in key D to confirm we've cleaned out the trash
gD = named.groupby(["repo_slug", "file_path", "scenario"]).size().sort_values(ascending=False)
print("\n=== top 10 longest under key D (cleaned) ===")
print(gD.head(10))

# Pick L_max from key E (the cleanest)
L_max_p95 = int(np.ceil(results["E_named_no_bg_len_ge_2"]["p95"]))
L_max_p99 = int(np.ceil(results["E_named_no_bg_len_ge_2"]["p99"]))
print(f"\n=== L_max recommendations (cleaned key E) ===")
print(f"  p95 -> L_max = {L_max_p95}")
print(f"  p99 -> L_max = {L_max_p99}")

# Persist
out = ANALYSIS_DIR / "scenario_identity_audit.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")
