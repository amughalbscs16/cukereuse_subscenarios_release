"""Eyeball the top patterns from Phase 2 to catch parser noise / generated suites.

A pattern that recurs 1,512 times in one file is either:
  - a real Background extraction candidate (rare at that scale),
  - a generated test artefact (Karate-style or property-based test
    generation),
  - a parser-noise accident.

Each top pattern gets printed with: text_seq, support, scope signals, and
a sample (repo, file, scenario) where it occurs.
"""

from __future__ import annotations

import pathlib
import textwrap

import pandas as pd

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"


def show(row: pd.Series, slices: pd.DataFrame, k: int = 3) -> None:
    print("-" * 78)
    print(f"L = {row['L']}, support_total = {row['support_total']:,}, "
          f"n_distinct_repos = {row['n_distinct_repos']}, "
          f"max_within_file_recurrence = {row['max_within_file_recurrence']:,}, "
          f"max_within_repo_files = {row['max_within_repo_files']:,}")
    print(f"cluster_id_seq: {list(row['cluster_id_seq'])}")
    print("canonical_text_seq:")
    for i, t in enumerate(row["canonical_text_seq"]):
        print(f"   [{i}] {textwrap.shorten(str(t), width=110)}")

    # Sample where it occurs
    sample = slices[slices["pattern"] == row["pattern"]].head(k)
    if len(sample):
        print(f"\nfirst {min(k, len(sample))} occurrences:")
        for _, s in sample.iterrows():
            print(f"   repo={s['repo_slug']:40s}  file={s['file_path']:50s}  "
                  f"scenario={textwrap.shorten(str(s['scenario']), 60)}")


def main() -> None:
    print("loading exact_subsequence_ranking.parquet ...")
    rank = pd.read_parquet(ANALYSIS_DIR / "exact_subsequence_ranking.parquet")

    print("loading slices.parquet (for one example per pattern) ...")
    slices = pd.read_parquet(ANALYSIS_DIR / "slices.parquet",
                             columns=["repo_slug", "file_path", "scenario",
                                      "cluster_id_seq"])
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]

    print("\n========================================================")
    print("TOP 5 BY support_total (raw recurrence)")
    print("========================================================")
    for _, r in rank.nlargest(5, "support_total").iterrows():
        show(r, slices)

    print("\n========================================================")
    print("TOP 5 within-file recurrence (RQ1: Background candidates)")
    print("========================================================")
    for _, r in rank.nlargest(5, "max_within_file_recurrence").iterrows():
        show(r, slices)

    print("\n========================================================")
    print("TOP 5 within-repo cross-file (RQ2: reusable scenario candidates)")
    print("========================================================")
    for _, r in rank.nlargest(5, "max_within_repo_files").iterrows():
        show(r, slices)

    print("\n========================================================")
    print("TOP 5 cross-repo (RQ3: shared higher-level step candidates)")
    print("========================================================")
    rq3_score = rank["n_distinct_repos"] * rank["L"]
    rank2 = rank.assign(_score=rq3_score)
    for _, r in rank2.nlargest(5, "_score").iterrows():
        show(r, slices)


if __name__ == "__main__":
    main()
