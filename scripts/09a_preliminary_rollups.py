"""Phase 9a — preliminary corpus-level rollups (pre-classifier).

Computes draft headline numbers for the abstract using all recurring
patterns, before any rubric labels exist. After Phase 6 (classifier)
lands we'll re-run with the "extraction-worthy" gate applied.

Headline candidates produced:
  - % of scenarios containing >= 1 recurring slice
  - % of files containing >= 1 within-file Background candidate
  - % of repos with >= 1 cross-file (RQ2) reusable-scenario candidate
  - % of repos contributing >= 1 cross-repo (RQ3) shared-step candidate
  - per-repo distribution: median / p25 / p75 of recurring slice count
  - per-mechanism counts: Background / reusable / shared
  - the "spec suite" outlier list — repos with >5x the median per-file
    Background candidate count, flagged for honest reporting

Inputs:  analysis/slices.parquet, analysis/exact_subsequence_ranking.parquet
Outputs: analysis/preliminary_rollups.json,
         analysis/spec_suite_outliers.csv
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"


def main() -> None:
    t0 = time.time()
    print("loading slices.parquet (key columns) ...")
    slices = pd.read_parquet(
        ANALYSIS_DIR / "slices.parquet",
        columns=["repo_slug", "file_path", "scenario", "cluster_id_seq", "L"],
    )
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]
    print(f"  {len(slices):,} slices")

    print("loading exact_subsequence_ranking.parquet ...")
    rank = pd.read_parquet(
        ANALYSIS_DIR / "exact_subsequence_ranking.parquet",
        columns=["pattern", "L", "support_total", "n_distinct_repos",
                 "n_distinct_orgs", "n_distinct_files",
                 "max_within_file_recurrence", "max_within_repo_files"],
    )
    print(f"  {len(rank):,} recurring patterns")

    # Compute per-pattern outlier fraction so we can report headline
    # numbers with and without spec-dominated patterns.
    print("\ncomputing per-pattern outlier fraction ...")
    outliers = pd.read_csv(ANALYSIS_DIR / "spec_suite_outliers.csv")
    outlier_keys = set(outliers["repo_slug"] + "|" + outliers["file_path"])
    slices["file_key"] = slices["repo_slug"] + "|" + slices["file_path"]
    slices["on_outlier"] = slices["file_key"].isin(outlier_keys)
    outlier_frac = (
        slices.groupby("pattern", sort=False)["on_outlier"].mean()
        .rename("outlier_fraction")
    )
    rank = rank.merge(outlier_frac, on="pattern", how="left")
    rank["outlier_fraction"] = rank["outlier_fraction"].fillna(0.0)
    real_signal_mask = rank["outlier_fraction"] <= 0.5
    print(f"  real-signal patterns (outlier_fraction <= 0.5):     {int(real_signal_mask.sum()):,}")
    print(f"  spec-dominated patterns (outlier_fraction > 0.5):  {int((~real_signal_mask).sum()):,}")

    # Three RQ masks (both views).
    # RQ3 uses n_distinct_orgs (Phase 2c correction): cross-organisational
    # recurrence, NOT the inflated cross-repo metric where one org's many
    # SDK repos counted as "5 distinct repos".
    def rq_pattern_sets(rank_df: pd.DataFrame) -> tuple[set, set, set]:
        return (
            set(rank_df.loc[rank_df["max_within_file_recurrence"] >= 2, "pattern"]),
            set(rank_df.loc[rank_df["max_within_repo_files"] >= 2, "pattern"]),
            set(rank_df.loc[rank_df["n_distinct_orgs"] >= 2, "pattern"]),
        )

    rq1_patterns, rq2_patterns, rq3_patterns = rq_pattern_sets(rank)
    rq1_real, rq2_real, rq3_real = rq_pattern_sets(rank[real_signal_mask])
    print(f"\n  full set:    RQ1={len(rq1_patterns):,}  RQ2={len(rq2_patterns):,}  RQ3={len(rq3_patterns):,}")
    print(f"  real-signal: RQ1={len(rq1_real):,}  RQ2={len(rq2_real):,}  RQ3={len(rq3_real):,}")

    # ---- corpus-level scenario prevalence ----
    t1 = time.time()
    print("\ncomputing scenario-level prevalence ...")
    n_scenarios_total = slices.groupby(
        ["repo_slug", "file_path", "scenario"], sort=False
    ).ngroups
    print(f"  total scenarios in mining set: {n_scenarios_total:,}")

    # A scenario "has a recurring slice" if it contains any pattern with
    # support_total >= 2. We can read this from the slices x rank join.
    recurring_patterns = set(rank["pattern"])
    slices["is_recurring"] = slices["pattern"].isin(recurring_patterns)
    has_recurring = (
        slices.groupby(["repo_slug", "file_path", "scenario"], sort=False)["is_recurring"]
        .any()
    )
    n_with_recurring = int(has_recurring.sum())
    pct_with_recurring = 100 * n_with_recurring / max(n_scenarios_total, 1)
    print(f"  scenarios with >=1 recurring slice: {n_with_recurring:,} "
          f"({pct_with_recurring:.1f}%)")

    # Per RQ (both views)
    slices["is_rq1"] = slices["pattern"].isin(rq1_patterns)
    slices["is_rq2"] = slices["pattern"].isin(rq2_patterns)
    slices["is_rq3"] = slices["pattern"].isin(rq3_patterns)
    slices["is_rq1_real"] = slices["pattern"].isin(rq1_real)
    slices["is_rq2_real"] = slices["pattern"].isin(rq2_real)
    slices["is_rq3_real"] = slices["pattern"].isin(rq3_real)

    def per_scen(col: str) -> pd.Series:
        return slices.groupby(
            ["repo_slug", "file_path", "scenario"], sort=False
        )[col].any()

    has_rq1 = per_scen("is_rq1"); has_rq2 = per_scen("is_rq2"); has_rq3 = per_scen("is_rq3")
    has_rq1_r = per_scen("is_rq1_real"); has_rq2_r = per_scen("is_rq2_real"); has_rq3_r = per_scen("is_rq3_real")

    def pct(s: pd.Series) -> float:
        return round(100 * float(s.sum()) / max(n_scenarios_total, 1), 2)

    print("\n  scenarios with >=1 RQ1 (Background) candidate:")
    print(f"     full:        {int(has_rq1.sum()):,}  ({pct(has_rq1)}%)")
    print(f"     real-signal: {int(has_rq1_r.sum()):,}  ({pct(has_rq1_r)}%)")
    print("  scenarios with >=1 RQ2 (reusable) candidate:")
    print(f"     full:        {int(has_rq2.sum()):,}  ({pct(has_rq2)}%)")
    print(f"     real-signal: {int(has_rq2_r.sum()):,}  ({pct(has_rq2_r)}%)")
    print("  scenarios with >=1 RQ3 (shared) candidate:")
    print(f"     full:        {int(has_rq3.sum()):,}  ({pct(has_rq3)}%)")
    print(f"     real-signal: {int(has_rq3_r.sum()):,}  ({pct(has_rq3_r)}%)")
    print(f"  ({time.time()-t1:.1f}s)")

    # ---- repo-level prevalence ----
    t2 = time.time()
    print("\ncomputing repo-level prevalence ...")
    n_repos_total = slices["repo_slug"].nunique()
    has_rq2_per_repo = slices.groupby("repo_slug", sort=False)["is_rq2"].any()
    has_rq3_per_repo = slices.groupby("repo_slug", sort=False)["is_rq3"].any()
    has_rq3_real_per_repo = slices.groupby("repo_slug", sort=False)["is_rq3_real"].any()
    print(f"  total repos in mining set: {n_repos_total}")
    print(f"  repos with >=1 RQ2 candidate: {int(has_rq2_per_repo.sum())} "
          f"({100*has_rq2_per_repo.sum()/n_repos_total:.1f}%)")
    print(f"  repos with >=1 RQ3 (cross-org) candidate (full): {int(has_rq3_per_repo.sum())} "
          f"({100*has_rq3_per_repo.sum()/n_repos_total:.1f}%)")
    print(f"  repos with >=1 RQ3 (cross-org) candidate (real-signal): {int(has_rq3_real_per_repo.sum())} "
          f"({100*has_rq3_real_per_repo.sum()/n_repos_total:.1f}%)")

    # Per-repo recurring-slice counts (distribution)
    per_repo_recurring = (
        slices[slices["is_recurring"]]
        .groupby("repo_slug", sort=False)["pattern"]
        .nunique()
    )
    print(f"  per-repo distinct recurring patterns: "
          f"median={int(per_repo_recurring.median())}, "
          f"p25={int(per_repo_recurring.quantile(0.25))}, "
          f"p75={int(per_repo_recurring.quantile(0.75))}, "
          f"max={int(per_repo_recurring.max())}")
    print(f"  ({time.time()-t2:.1f}s)")

    # ---- file-level RQ1 prevalence ----
    t3 = time.time()
    print("\ncomputing file-level RQ1 (Background candidate) prevalence ...")
    n_files_total = slices.groupby(["repo_slug", "file_path"], sort=False).ngroups
    has_rq1_per_file = (
        slices.groupby(["repo_slug", "file_path"], sort=False)["is_rq1"].any()
    )
    print(f"  total .feature files in mining set: {n_files_total:,}")
    print(f"  files with >=1 RQ1 candidate: {int(has_rq1_per_file.sum()):,} "
          f"({100*has_rq1_per_file.sum()/n_files_total:.1f}%)")
    print(f"  ({time.time()-t3:.1f}s)")

    # ---- spec-suite outlier reporting ----
    # Phase 2c (v3 detector: density AND (top-recurrence OR template-heavy))
    # is authoritative for the outlier list. We report against that file
    # without overwriting it.
    t4 = time.time()
    print("\nreporting spec-suite outliers (Phase 2c v3 detector) ...")
    rq1_per_file = (
        slices[slices["is_rq1"]]
        .groupby(["repo_slug", "file_path"], sort=False)["pattern"]
        .nunique()
    )
    median_per_file = float(rq1_per_file.median()) if len(rq1_per_file) else 0.0
    n_outlier_files_v3 = int(len(outliers))  # already loaded at top from csv
    outlier_repo_counts = (
        outliers.groupby("repo_slug", sort=False).size()
        .sort_values(ascending=False)
    )
    print(f"  median per-file RQ1 patterns: {median_per_file:.1f}")
    print(f"  v3 outlier files (from spec_suite_outliers.csv): {n_outlier_files_v3:,}")
    print(f"  top outlier repos by # of v3 outlier files:")
    for repo, n in outlier_repo_counts.head(10).items():
        print(f"     {repo:60s}  {n} files flagged")
    print(f"  ({time.time()-t4:.1f}s)")

    # ---- write summary ----
    summary = {
        "n_slices_total": int(len(slices)),
        "n_recurring_patterns": int(len(rank)),
        "n_real_signal_patterns": int(real_signal_mask.sum()),
        "n_spec_dominated_patterns": int((~real_signal_mask).sum()),
        "scenario_level_full": {
            "n_scenarios_total": int(n_scenarios_total),
            "n_with_any_recurring": int(n_with_recurring),
            "pct_with_any_recurring": round(pct_with_recurring, 2),
            "n_with_rq1": int(has_rq1.sum()),
            "pct_with_rq1": pct(has_rq1),
            "n_with_rq2": int(has_rq2.sum()),
            "pct_with_rq2": pct(has_rq2),
            "n_with_rq3": int(has_rq3.sum()),
            "pct_with_rq3": pct(has_rq3),
        },
        "scenario_level_real_signal": {
            "n_with_rq1": int(has_rq1_r.sum()),
            "pct_with_rq1": pct(has_rq1_r),
            "n_with_rq2": int(has_rq2_r.sum()),
            "pct_with_rq2": pct(has_rq2_r),
            "n_with_rq3": int(has_rq3_r.sum()),
            "pct_with_rq3": pct(has_rq3_r),
        },
        "repo_level": {
            "n_repos_total": int(n_repos_total),
            "n_with_rq2_candidate": int(has_rq2_per_repo.sum()),
            "pct_with_rq2_candidate": round(100 * has_rq2_per_repo.sum() / n_repos_total, 2),
            "n_with_rq3_candidate": int(has_rq3_per_repo.sum()),
            "pct_with_rq3_candidate": round(100 * has_rq3_per_repo.sum() / n_repos_total, 2),
            "per_repo_distinct_recurring_patterns": {
                "median": int(per_repo_recurring.median()),
                "p25": int(per_repo_recurring.quantile(0.25)),
                "p75": int(per_repo_recurring.quantile(0.75)),
                "p95": int(per_repo_recurring.quantile(0.95)),
                "max": int(per_repo_recurring.max()),
            },
        },
        "file_level": {
            "n_files_total": int(n_files_total),
            "n_with_rq1_candidate": int(has_rq1_per_file.sum()),
            "pct_with_rq1_candidate": round(100 * has_rq1_per_file.sum() / n_files_total, 2),
        },
        "spec_suite_outliers": {
            "median_per_file_rq1_patterns": median_per_file,
            "detector_version": "v3 (Phase 2c: density AND (top-recurrence OR template-heavy))",
            "n_outlier_files": n_outlier_files_v3,
            "top_outlier_repos": outlier_repo_counts.head(10).to_dict(),
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_path = ANALYSIS_DIR / "preliminary_rollups.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    print(f"\ntotal wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
