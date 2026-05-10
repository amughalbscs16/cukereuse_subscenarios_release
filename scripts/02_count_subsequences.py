"""Phase 2 — exact subsequence counting across three scopes.

For each unique cluster_id_seq pattern from analysis/slices.parquet,
compute per-pattern aggregates that distinguish the three RQs:

  RQ1 (Background candidate):       max_within_file_recurrence
       = max #distinct-scenarios within any single (repo, file) containing the pattern.
       A pattern with this >= 2 means at least two scenarios in one file
       share the slice, i.e., a within-file Background candidate.

  RQ2 (reusable-scenario candidate): max_within_repo_files
       = max #distinct-files within any single repo containing the pattern.
       A pattern with this >= 2 means a single repo has the slice in
       multiple files — a reusable-scenario candidate per Mughal 2024.

  RQ3 (shared higher-level step):    n_distinct_repos
       = number of distinct repos containing the pattern.
       A pattern with this >= 2 means cross-organisational recurrence — a
       shared higher-level step candidate per Mughal 2024.

Output: analysis/exact_subsequence_ranking.parquet with columns
    pattern, L, canonical_text_seq, support_total,
    n_distinct_repos, n_distinct_files, n_distinct_scenarios,
    max_within_file_recurrence, max_within_repo_files

Patterns with support_total < 2 are dropped (they cannot recur).
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
    in_path = ANALYSIS_DIR / "slices.parquet"
    print(f"loading {in_path} ...")
    slices = pd.read_parquet(in_path)
    print(f"  rows: {len(slices):,}  ({time.time()-t0:.1f}s)")

    # Build pattern key (string) — vectorised list comp
    t1 = time.time()
    print("building pattern keys ...")
    patterns = [",".join(map(str, seq)) for seq in slices["cluster_id_seq"]]
    slices["pattern"] = patterns
    print(f"  built {len(patterns):,} pattern strings ({time.time()-t1:.1f}s)")

    # Composite keys for distinct counts
    t2 = time.time()
    print("building composite keys ...")
    slices["file_key"] = slices["repo_slug"] + "|" + slices["file_path"]
    slices["scenario_key"] = slices["file_key"] + "|" + slices["scenario"]
    print(f"  done ({time.time()-t2:.1f}s)")

    # First-pass aggregates per pattern
    t3 = time.time()
    print("aggregating per-pattern ...")
    agg = slices.groupby("pattern", sort=False).agg(
        L=("L", "first"),
        support_total=("slice_id", "count"),
        n_distinct_repos=("repo_slug", "nunique"),
        n_distinct_files=("file_key", "nunique"),
        n_distinct_scenarios=("scenario_key", "nunique"),
    )
    print(f"  unique patterns: {len(agg):,}  ({time.time()-t3:.1f}s)")

    # max_within_file_recurrence per pattern
    t4 = time.time()
    print("computing max_within_file_recurrence ...")
    within_file = (
        slices.groupby(["pattern", "file_key"], sort=False)["scenario_key"]
        .nunique()
        .groupby(level=0).max()
        .rename("max_within_file_recurrence")
    )
    print(f"  done ({time.time()-t4:.1f}s)")

    # max_within_repo_files per pattern
    t5 = time.time()
    print("computing max_within_repo_files ...")
    within_repo = (
        slices.groupby(["pattern", "repo_slug"], sort=False)["file_key"]
        .nunique()
        .groupby(level=0).max()
        .rename("max_within_repo_files")
    )
    print(f"  done ({time.time()-t5:.1f}s)")

    # Canonical text_seq per pattern (first occurrence)
    t6 = time.time()
    print("attaching canonical text_seq ...")
    canonical = (
        slices.drop_duplicates("pattern", keep="first")
        .set_index("pattern")["text_seq"]
        .rename("canonical_text_seq")
    )
    print(f"  done ({time.time()-t6:.1f}s)")

    # Join all
    t7 = time.time()
    print("joining ...")
    result = (
        agg.join(within_file).join(within_repo).join(canonical).reset_index()
    )
    # cluster_id_seq tuple form for downstream readability
    result["cluster_id_seq"] = result["pattern"].apply(
        lambda s: [int(x) for x in s.split(",")]
    )
    # Drop singletons
    n_before = len(result)
    result = result[result["support_total"] >= 2]
    print(f"  rows after support>=2 filter: {len(result):,} (dropped {n_before-len(result):,})  "
          f"({time.time()-t7:.1f}s)")

    # Reorder columns
    result = result[[
        "pattern", "cluster_id_seq", "L", "canonical_text_seq",
        "support_total", "n_distinct_repos", "n_distinct_files",
        "n_distinct_scenarios", "max_within_file_recurrence",
        "max_within_repo_files",
    ]]

    # Write
    t8 = time.time()
    out_path = ANALYSIS_DIR / "exact_subsequence_ranking.parquet"
    print(f"writing {out_path} ...")
    result.to_parquet(out_path, compression="zstd", index=False)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {file_mb:.1f} MB  ({time.time()-t8:.1f}s)")

    # Top-line stats
    print("\n=== top-line stats ===")
    print(f"  unique recurring patterns:       {len(result):,}")
    print(f"  with cross-repo (RQ3) signal >=2: {(result['n_distinct_repos']>=2).sum():,}")
    print(f"  with within-repo (RQ2) signal >=2: {(result['max_within_repo_files']>=2).sum():,}")
    print(f"  with within-file (RQ1) signal >=2: {(result['max_within_file_recurrence']>=2).sum():,}")
    print(f"  median L: {int(result['L'].median())}")
    print(f"  median support_total: {int(result['support_total'].median())}")

    print("\n=== top 5 by support_total ===")
    print(result.nlargest(5, "support_total")[
        ["L", "support_total", "n_distinct_repos",
         "max_within_file_recurrence", "max_within_repo_files"]
    ].to_string())

    print("\n=== top 5 cross-repo (RQ3) candidates by n_distinct_repos*L ===")
    rq3 = result.assign(score_rq3=result["n_distinct_repos"] * result["L"])
    print(rq3.nlargest(5, "score_rq3")[
        ["L", "support_total", "n_distinct_repos", "score_rq3"]
    ].to_string())

    print("\n=== top 5 within-repo (RQ2) candidates by max_within_repo_files*L ===")
    rq2 = result.assign(score_rq2=result["max_within_repo_files"] * result["L"])
    print(rq2.nlargest(5, "score_rq2")[
        ["L", "support_total", "max_within_repo_files", "score_rq2"]
    ].to_string())

    print("\n=== top 5 within-file (RQ1) candidates by max_within_file_recurrence*L ===")
    rq1 = result.assign(score_rq1=result["max_within_file_recurrence"] * result["L"])
    print(rq1.nlargest(5, "score_rq1")[
        ["L", "support_total", "max_within_file_recurrence", "score_rq1"]
    ].to_string())

    summary = {
        "n_slices_input": int(len(slices)),
        "n_unique_patterns": int(len(agg)),
        "n_recurring_patterns": int(len(result)),
        "rq1_within_file_candidates": int((result["max_within_file_recurrence"] >= 2).sum()),
        "rq2_within_repo_candidates": int((result["max_within_repo_files"] >= 2).sum()),
        "rq3_cross_repo_candidates": int((result["n_distinct_repos"] >= 2).sum()),
        "wall_seconds": round(time.time() - t0, 1),
        "output_mb": round(file_mb, 1),
    }
    summary_path = ANALYSIS_DIR / "exact_subsequence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")
    print(f"\ntotal wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
