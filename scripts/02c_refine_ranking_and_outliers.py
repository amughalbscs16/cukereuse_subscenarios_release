"""Phase 2c — refine the ranking with two corrections from pilot labelling.

CORRECTION 1 — same-org cross-repo inflation.
================================================
Phase 2 counts n_distinct_repos. But several patterns get high RQ3 signals
because one organisation publishes many repos (e.g., DataDog ships
datadog-api-client-go / -python / -java / ... and the test corpus picks
all of them up as "5 distinct repos"). For RQ3 ("shared higher-level
step across organisations"), we want cross-org recurrence, not
cross-language-client recurrence.

Adds:
  - n_distinct_orgs per pattern: distinct first-segment of repo_slug
    (the GitHub owner). E.g., DataDog_datadog-api-client-go -> "DataDog".

CORRECTION 2 — spec-suite detector is too broad.
================================================
The current outlier list flags any file with > 50 distinct RQ1 patterns,
which catches both (a) truly-generated suites (e.g., AWS service specs
like local-web-services memorydb/neptune) and (b) legitimately
heavily-duplicated test code (e.g., Corvusoft restq's HTTP header
negative-assertion patterns). Pilot labelling Entry #9 confirmed (b) is
a real extraction candidate, not noise.

Tightened detector (v2):
  A file is a "spec suite" iff BOTH
    (A) >= 50 distinct RQ1 patterns                   [original density]
    (B) >= 50% of those patterns' canonical texts contain the regex
        `"\\w+"\\s+"\\w+"` (two adjacent quoted single-words),
        which is the template-placeholder signature of generated specs
        (e.g., `every active "memorydb" "cluster" has ...`).

This re-classifies files like Corvusoft restq as "high-recurrence real"
rather than spec-noise.

Outputs:
  - exact_subsequence_ranking.parquet (overwritten with n_distinct_orgs)
  - spec_suite_outliers.csv (overwritten — tightened list)
  - phase2c_summary.json
"""

from __future__ import annotations

import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"

TEMPLATE_RE = re.compile(r'"\w+"\s+"\w+"')  # two adjacent quoted single-words


def main() -> None:
    t0 = time.time()

    # ---- correction 1: add n_distinct_orgs ----
    print("loading slices.parquet (key columns) ...")
    slices = pd.read_parquet(
        ANALYSIS_DIR / "slices.parquet",
        columns=["repo_slug", "file_path", "scenario", "cluster_id_seq"],
    )
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]
    slices["org"] = slices["repo_slug"].str.split("_", n=1).str[0]
    print(f"  {len(slices):,} slices  ({time.time()-t0:.1f}s)")

    print("\nloading exact_subsequence_ranking.parquet ...")
    rank = pd.read_parquet(ANALYSIS_DIR / "exact_subsequence_ranking.parquet")
    print(f"  {len(rank):,} patterns")

    print("computing n_distinct_orgs per pattern ...")
    t1 = time.time()
    # Drop columns we'll re-add to avoid merge suffix collisions on re-run
    for col in ("n_distinct_orgs", "has_template_structure"):
        if col in rank.columns:
            rank = rank.drop(columns=col)
    org_counts = (
        slices.groupby("pattern", sort=False)["org"]
        .nunique()
        .rename("n_distinct_orgs")
        .reset_index()
    )
    rank = rank.merge(org_counts, on="pattern", how="left")
    rank["n_distinct_orgs"] = rank["n_distinct_orgs"].fillna(1).astype(np.int32)
    n_rq3_old = int((rank["n_distinct_repos"] >= 2).sum())
    n_rq3_new = int((rank["n_distinct_orgs"] >= 2).sum())
    print(f"  RQ3 candidates by n_distinct_repos>=2:  {n_rq3_old:,}")
    print(f"  RQ3 candidates by n_distinct_orgs>=2:   {n_rq3_new:,}  "
          f"(dropped {n_rq3_old - n_rq3_new:,} same-org cross-repo cases)")
    print(f"  ({time.time()-t1:.1f}s)")

    # ---- correction 2: tighten spec-suite detector ----
    print("\ntightening spec-suite detector (file density + template cue) ...")
    t2 = time.time()
    # Per pattern: does the canonical text contain template-pattern regex?
    # Use any step in the slice; if any step matches, the slice is template-y.
    canonical = rank["canonical_text_seq"].apply(
        lambda seq: any(TEMPLATE_RE.search(str(s) or "") for s in seq)
    )
    rank["has_template_structure"] = canonical
    n_template = int(canonical.sum())
    print(f"  patterns with template-structure cue: {n_template:,} "
          f"({100*n_template/len(rank):.1f}%)")

    # Compute per-file: max recurrence of any single pattern in this file.
    # Generated suites are characterised by ONE pattern recurring 100s of
    # times in a single file (a generator emits the same template-step in
    # every test instance). Legit-duplicated code rarely exceeds ~30.
    print("\n  computing per-file max single-pattern recurrence ...")
    rq1_mask = rank["max_within_file_recurrence"] >= 2
    rq1_patterns_set = set(rank.loc[rq1_mask, "pattern"])

    slices["is_rq1_pattern"] = slices["pattern"].isin(rq1_patterns_set)
    rq1_slices = slices[slices["is_rq1_pattern"]]

    # For each (repo, file, pattern), count occurrences (= # distinct
    # scenarios that contain this pattern in this file).
    # max over patterns -> file's "top recurrence"
    per_pat_in_file = (
        rq1_slices.groupby(
            ["repo_slug", "file_path", "scenario", "pattern"], sort=False
        ).size().reset_index().groupby(
            ["repo_slug", "file_path", "pattern"], sort=False
        ).size().reset_index().rename(columns={0: "n_scen_with_pattern"})
    )
    file_top_recur = (
        per_pat_in_file.groupby(["repo_slug", "file_path"], sort=False)
        ["n_scen_with_pattern"].max()
        .rename("top_pattern_recurrence")
    )

    file_stats = file_groups_dummy = (
        rq1_slices.groupby(["repo_slug", "file_path"], sort=False)
        .agg(n_distinct_rq1_patterns=("pattern", "nunique"))
    )
    file_stats = file_stats.join(file_top_recur)
    file_stats["template_fraction"] = (
        rq1_slices.groupby(["repo_slug", "file_path"], sort=False)["pattern"]
        .apply(lambda s: rank.set_index("pattern")
               .loc[s.unique(), "has_template_structure"].mean())
    )

    DENSITY_THRESHOLD = 50
    TOP_RECUR_THRESHOLD = 100        # generated-suite cue (one pattern 100+ times in one file)
    TEMPLATE_THRESHOLD = 0.30        # at least 30% of patterns template-structured

    is_density = file_stats["n_distinct_rq1_patterns"] > DENSITY_THRESHOLD
    is_top_recur = file_stats["top_pattern_recurrence"] > TOP_RECUR_THRESHOLD
    is_template_heavy = file_stats["template_fraction"] >= TEMPLATE_THRESHOLD

    # v3 spec-suite: (density AND top-recurrence) OR (density AND template-heavy)
    file_stats["is_spec_suite_v3"] = is_density & (is_top_recur | is_template_heavy)
    print(f"  density-flagged files (>{DENSITY_THRESHOLD} RQ1 patterns):  "
          f"{int(is_density.sum()):,}")
    print(f"  top-recurrence-flagged files (>{TOP_RECUR_THRESHOLD} per-file): "
          f"{int(is_top_recur.sum()):,}")
    print(f"  template-heavy files (>={TEMPLATE_THRESHOLD} fraction):       "
          f"{int(is_template_heavy.sum()):,}")
    print(f"  spec-suite (v3) = density AND (top-recur OR template):       "
          f"{int(file_stats['is_spec_suite_v3'].sum()):,}")
    print(f"  ({time.time()-t2:.1f}s)")

    # Save new outlier list
    outliers_v3 = (
        file_stats[file_stats["is_spec_suite_v3"]]
        .reset_index()
        .sort_values("top_pattern_recurrence", ascending=False)
    )
    outliers_v3_path = ANALYSIS_DIR / "spec_suite_outliers.csv"
    outliers_v3.to_csv(outliers_v3_path, index=False)
    print(f"  wrote tightened outlier list to {outliers_v3_path} "
          f"({len(outliers_v3):,} files)")

    # Top outlier repos under tightened detector
    top_repos = outliers_v3.groupby("repo_slug", sort=False).size().sort_values(ascending=False)
    print(f"\n  top spec-suite-v3 repos:")
    for repo, n in top_repos.head(15).items():
        print(f"     {repo:60s}  {n} files")

    # Save augmented ranking
    rank.to_parquet(
        ANALYSIS_DIR / "exact_subsequence_ranking.parquet",
        compression="zstd", index=False,
    )
    print(f"\nupdated exact_subsequence_ranking.parquet (added n_distinct_orgs, has_template_structure)")

    # Summary
    summary = {
        "n_patterns": int(len(rank)),
        "rq3_repos_old": n_rq3_old,
        "rq3_orgs_new": n_rq3_new,
        "rq3_dropped_same_org_inflation": n_rq3_old - n_rq3_new,
        "n_patterns_with_template_structure": n_template,
        "spec_suite_v3_files": int(file_stats["is_spec_suite_v3"].sum()),
        "spec_suite_v1_files_dropped": int(is_density.sum() - file_stats["is_spec_suite_v3"].sum()),
        "v3_thresholds": {
            "density": DENSITY_THRESHOLD,
            "top_pattern_recurrence": TOP_RECUR_THRESHOLD,
            "template_fraction": TEMPLATE_THRESHOLD,
            "rule": "density AND (top-recurrence OR template-heavy)",
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    (ANALYSIS_DIR / "phase2c_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
