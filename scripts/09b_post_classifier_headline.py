"""Phase 9b — post-classifier corpus-level headline.

Apply the Phase-6 extraction-worthy classifier as a gate and recompute
the scenario-level / repo-level / file-level prevalence numbers, this
time restricted to patterns the classifier predicts are extraction-worthy.

Inputs:
  analysis/slices.parquet
  analysis/extraction_classifier_predictions.parquet

Output:
  analysis/post_classifier_headline.json
"""

from __future__ import annotations

import json
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"


def main() -> None:
    t0 = time.time()
    print("loading classifier predictions ...")
    preds = pd.read_parquet(ANALYSIS / "extraction_classifier_predictions.parquet")
    print(f"  {len(preds):,} scope-eligible patterns")
    print(f"  {int(preds['pred_extraction_worthy'].sum()):,} predicted extraction-worthy")

    print("\nloading slices ...")
    slices = pd.read_parquet(
        ANALYSIS / "slices.parquet",
        columns=["repo_slug", "file_path", "scenario", "cluster_id_seq"],
    )
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]
    print(f"  {len(slices):,} slices")

    # Three RQ pattern sets restricted to predicted extraction-worthy
    ew = preds[preds["pred_extraction_worthy"] == 1].copy()
    rq1_ew = set(ew.loc[ew["max_within_file_recurrence"] >= 2, "pattern"])
    rq2_ew = set(ew.loc[ew["max_within_repo_files"] >= 2, "pattern"])
    rq3_ew = set(ew.loc[ew["n_distinct_orgs"] >= 2, "pattern"])
    print(f"\n  predicted-EW with RQ1 signal: {len(rq1_ew):,}")
    print(f"  predicted-EW with RQ2 signal: {len(rq2_ew):,}")
    print(f"  predicted-EW with RQ3 signal: {len(rq3_ew):,}")

    slices["is_rq1_ew"] = slices["pattern"].isin(rq1_ew)
    slices["is_rq2_ew"] = slices["pattern"].isin(rq2_ew)
    slices["is_rq3_ew"] = slices["pattern"].isin(rq3_ew)

    n_scenarios = slices.groupby(["repo_slug", "file_path", "scenario"], sort=False).ngroups
    print(f"\n  total scenarios: {n_scenarios:,}")

    def per_scen(col):
        return slices.groupby(["repo_slug", "file_path", "scenario"], sort=False)[col].any()

    has_rq1_ew = per_scen("is_rq1_ew")
    has_rq2_ew = per_scen("is_rq2_ew")
    has_rq3_ew = per_scen("is_rq3_ew")

    def pct(s):
        return round(100 * float(s.sum()) / max(n_scenarios, 1), 2)

    print("\n  scenarios with >=1 EXTRACTION-WORTHY recurring slice:")
    print(f"    RQ1 (Background candidate):       {int(has_rq1_ew.sum()):,}  ({pct(has_rq1_ew)}%)")
    print(f"    RQ2 (reusable scenario candidate): {int(has_rq2_ew.sum()):,}  ({pct(has_rq2_ew)}%)")
    print(f"    RQ3 (shared step candidate):       {int(has_rq3_ew.sum()):,}  ({pct(has_rq3_ew)}%)")

    n_repos = slices["repo_slug"].nunique()
    has_rq2_per_repo = slices.groupby("repo_slug", sort=False)["is_rq2_ew"].any()
    has_rq3_per_repo = slices.groupby("repo_slug", sort=False)["is_rq3_ew"].any()
    print(f"\n  total repos: {n_repos}")
    print(f"  repos with >=1 EW RQ2 candidate: {int(has_rq2_per_repo.sum())} "
          f"({100*has_rq2_per_repo.sum()/n_repos:.1f}%)")
    print(f"  repos with >=1 EW RQ3 candidate: {int(has_rq3_per_repo.sum())} "
          f"({100*has_rq3_per_repo.sum()/n_repos:.1f}%)")

    summary = {
        "n_predicted_extraction_worthy_patterns": int(ew["pred_extraction_worthy"].sum()),
        "n_predicted_ew_with_rq1": len(rq1_ew),
        "n_predicted_ew_with_rq2": len(rq2_ew),
        "n_predicted_ew_with_rq3": len(rq3_ew),
        "scenario_level_extraction_worthy": {
            "n_scenarios_total": int(n_scenarios),
            "n_with_rq1_ew": int(has_rq1_ew.sum()),
            "pct_with_rq1_ew": pct(has_rq1_ew),
            "n_with_rq2_ew": int(has_rq2_ew.sum()),
            "pct_with_rq2_ew": pct(has_rq2_ew),
            "n_with_rq3_ew": int(has_rq3_ew.sum()),
            "pct_with_rq3_ew": pct(has_rq3_ew),
        },
        "repo_level_extraction_worthy": {
            "n_repos_total": int(n_repos),
            "n_with_rq2_ew_candidate": int(has_rq2_per_repo.sum()),
            "pct_with_rq2_ew_candidate": round(100*has_rq2_per_repo.sum()/n_repos, 2),
            "n_with_rq3_ew_candidate": int(has_rq3_per_repo.sum()),
            "pct_with_rq3_ew_candidate": round(100*has_rq3_per_repo.sum()/n_repos, 2),
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = ANALYSIS / "post_classifier_headline.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
