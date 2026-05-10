"""Phase 5a — build the stratified labelling pool.

Per the rubric (methodology/LABELING_RUBRIC_paper2_slice_level.md sec.4),
sample N=200 patterns stratified by:
  - L bucket: {2,3}, {4,5}, {6,7,8}, {9..12}, {13..18}    (5 strata)
  - scope:    RQ1 (within-file), RQ2 (within-repo), RQ3 (cross-repo)
              (3 strata; multi-scope patterns assigned to the most-
               specific RQ they qualify for: RQ3 > RQ2 > RQ1)
  - support:  {2..5}, {6..20}, {21..100}, {>100}           (4 strata)
  -> 5*3*4 = 60 cells; sample <=4 per cell, with proportional
     re-balance for empty cells (target N=200).

Plus: 60-slice overlap subset (labelled by all three authors) for
Fleiss-kappa. The remaining 140 split 70/35/35 across authors.

Each pattern in the pool gets:
  - canonical text_seq (presented to labeller)
  - scope signals (max_within_file_recurrence, max_within_repo_files,
    n_distinct_repos)
  - one representative occurrence (repo, file, scenario)
  - is_outlier flag (from spec_suite_outliers.csv, for transparency)

Output: methodology/labeling_pool.jsonl   (one line per slice)
        methodology/labeling_pool_stats.json
"""

from __future__ import annotations

import json
import pathlib
import random

import pandas as pd

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"
METHODOLOGY_DIR = pathlib.Path(__file__).resolve().parent.parent / "methodology"
METHODOLOGY_DIR.mkdir(exist_ok=True)

L_BUCKETS = [(2, 3, "L_2_3"), (4, 5, "L_4_5"), (6, 8, "L_6_8"),
             (9, 12, "L_9_12"), (13, 18, "L_13_18")]
SUPPORT_BUCKETS = [(2, 5, "sup_2_5"), (6, 20, "sup_6_20"),
                   (21, 100, "sup_21_100"), (101, 10**9, "sup_gt100")]
N_TARGET = 200
N_OVERLAP = 60
PER_CELL_CAP = 4
SEED = 42


def L_bucket(L: int) -> str:
    for lo, hi, name in L_BUCKETS:
        if lo <= L <= hi:
            return name
    raise ValueError(f"out-of-range L={L}")


def support_bucket(s: int) -> str:
    for lo, hi, name in SUPPORT_BUCKETS:
        if lo <= s <= hi:
            return name
    raise ValueError(f"out-of-range support={s}")


def scope_label(row: pd.Series) -> str:
    # Most-specific scope wins
    if row["n_distinct_repos"] >= 2:
        return "RQ3"
    if row["max_within_repo_files"] >= 2:
        return "RQ2"
    if row["max_within_file_recurrence"] >= 2:
        return "RQ1"
    return "noscope"


def main() -> None:
    print("loading exact_subsequence_ranking.parquet ...")
    rank = pd.read_parquet(ANALYSIS_DIR / "exact_subsequence_ranking.parquet")
    print(f"  {len(rank):,} recurring patterns")

    rank["L_bucket"] = rank["L"].apply(L_bucket)
    rank["sup_bucket"] = rank["support_total"].apply(support_bucket)
    rank["scope"] = rank.apply(scope_label, axis=1)
    rank = rank[rank["scope"] != "noscope"].reset_index(drop=True)
    print(f"  patterns with at least one scope signal: {len(rank):,}")

    # Outlier flag (from spec-suite detector)
    outliers = pd.read_csv(ANALYSIS_DIR / "spec_suite_outliers.csv")
    outlier_keys = set(outliers["repo_slug"] + "|" + outliers["file_path"])

    # Compute per-pattern outlier fraction:
    #   For each pattern, what fraction of its slice occurrences are on
    #   spec-suite-outlier files? Patterns where this > 0.5 are dominated
    #   by generated suites and should be down-weighted (or limited to a
    #   small spec-suite stratum) in the labelling pool.
    print("\ncomputing per-pattern outlier fraction ...")
    slices_full = pd.read_parquet(
        ANALYSIS_DIR / "slices.parquet",
        columns=["repo_slug", "file_path", "scenario", "cluster_id_seq"],
    )
    slices_full["pattern"] = [",".join(map(str, s)) for s in slices_full["cluster_id_seq"]]
    slices_full["file_key"] = slices_full["repo_slug"] + "|" + slices_full["file_path"]
    slices_full["on_outlier"] = slices_full["file_key"].isin(outlier_keys)
    outlier_frac = (
        slices_full.groupby("pattern", sort=False)["on_outlier"]
        .mean()
        .rename("outlier_fraction")
    )
    rank = rank.merge(outlier_frac, on="pattern", how="left")
    rank["outlier_fraction"] = rank["outlier_fraction"].fillna(0.0)
    n_real = int((rank["outlier_fraction"] <= 0.5).sum())
    n_spec = int((rank["outlier_fraction"] > 0.5).sum())
    print(f"  real-signal patterns (outlier_fraction <= 0.5):     {n_real:,}")
    print(f"  spec-dominated patterns (outlier_fraction > 0.5):  {n_spec:,}")

    # Restrict main pool to real-signal patterns. We'll separately sample
    # 20 from spec-dominated for rubric coverage of the flagged-spec case.
    rank_real = rank[rank["outlier_fraction"] <= 0.5].reset_index(drop=True)
    rank_spec = rank[rank["outlier_fraction"] > 0.5].reset_index(drop=True)
    print(f"  main pool (real-signal): {len(rank_real):,}")
    print(f"  spec-coverage pool:       {len(rank_spec):,}")

    def stratified_sample(df: pd.DataFrame, target: int, rng: random.Random) -> list[int]:
        cells = (
            df.groupby(["L_bucket", "scope", "sup_bucket"], sort=False)
            .size().rename("n_in_cell").reset_index()
        )
        cap = max(1, target // max(len(cells), 1))
        picks: list[int] = []
        cell_picks = {}
        for _, c in cells.iterrows():
            sub = df[
                (df["L_bucket"] == c["L_bucket"]) &
                (df["scope"] == c["scope"]) &
                (df["sup_bucket"] == c["sup_bucket"])
            ]
            n_pick = min(cap, len(sub))
            idxs = rng.sample(list(sub.index), n_pick)
            cell_picks[(c["L_bucket"], c["scope"], c["sup_bucket"])] = idxs
            picks.extend(idxs)
        # Top-up
        if len(picks) < target:
            deficit = target - len(picks)
            leftovers = []
            for (lb, sc, sb), idxs in cell_picks.items():
                sub = df[
                    (df["L_bucket"] == lb) & (df["scope"] == sc) & (df["sup_bucket"] == sb)
                ]
                leftovers.extend([i for i in sub.index if i not in idxs])
            rng.shuffle(leftovers)
            picks.extend(leftovers[:deficit])
        elif len(picks) > target:
            rng.shuffle(picks)
            picks = picks[:target]
        return list(dict.fromkeys(picks))[:target]

    # Sample 180 from real-signal, 20 from spec-dominated (= 200 total)
    rng = random.Random(SEED)
    real_keys = stratified_sample(rank_real, 180, rng)
    spec_keys = stratified_sample(rank_spec, 20, rng)

    pool_df = pd.concat(
        [rank_real.loc[real_keys].assign(stratum="real_signal"),
         rank_spec.loc[spec_keys].assign(stratum="spec_dominated")],
        ignore_index=True,
    )
    print(f"\nfinal pool size: {len(pool_df):,} "
          f"(real_signal={len(real_keys)}, spec_dominated={len(spec_keys)})")

    # Pull a representative occurrence — prefer non-outlier file for
    # real-signal patterns (so labellers see a real example), and
    # accept any file for spec-dominated patterns.
    print("\nattaching representative occurrences (non-outlier preferred for real-signal) ...")
    slices_full["non_outlier_priority"] = (~slices_full["on_outlier"]).astype(int)
    slices_sorted = slices_full.sort_values(
        ["pattern", "non_outlier_priority"], ascending=[True, False], kind="stable"
    )
    occ = (
        slices_sorted.drop_duplicates("pattern", keep="first")
        .set_index("pattern")[["repo_slug", "file_path", "scenario"]]
    )
    pool_df = pool_df.join(occ, on="pattern")

    # Outlier flag on the chosen example (sanity check)
    pool_df["example_on_outlier_file"] = (
        (pool_df["repo_slug"] + "|" + pool_df["file_path"]).isin(outlier_keys)
    )
    print(f"  pool entries whose example is on an outlier file: "
          f"{int(pool_df['example_on_outlier_file'].sum())} "
          f"(expected: ~20 from spec_dominated stratum)")

    # Overlap and split assignment
    rng2 = random.Random(SEED + 1)
    indices = list(range(len(pool_df)))
    rng2.shuffle(indices)
    overlap_idx = set(indices[:N_OVERLAP])
    rest = indices[N_OVERLAP:]
    n_each = len(rest) // 3
    a, b, c = rest[:n_each], rest[n_each:2*n_each], rest[2*n_each:]

    assignment = ["overlap"] * len(pool_df)
    for i in a: assignment[i] = "author_A"
    for i in b: assignment[i] = "author_B"
    for i in c: assignment[i] = "author_C"
    pool_df["assignment"] = assignment

    # Write JSONL
    out_jsonl = METHODOLOGY_DIR / "labeling_pool.jsonl"
    print(f"\nwriting {out_jsonl} ...")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, r in pool_df.iterrows():
            rec = {
                "pattern": r["pattern"],
                "L": int(r["L"]),
                "L_bucket": r["L_bucket"],
                "support_total": int(r["support_total"]),
                "support_bucket": r["sup_bucket"],
                "scope": r["scope"],
                "stratum": r["stratum"],
                "outlier_fraction": round(float(r["outlier_fraction"]), 3),
                "n_distinct_repos": int(r["n_distinct_repos"]),
                "n_distinct_files": int(r["n_distinct_files"]),
                "max_within_file_recurrence": int(r["max_within_file_recurrence"]),
                "max_within_repo_files": int(r["max_within_repo_files"]),
                "canonical_text_seq": list(r["canonical_text_seq"]),
                "example_repo": r["repo_slug"],
                "example_file": r["file_path"],
                "example_scenario": r["scenario"],
                "example_on_outlier_file": bool(r["example_on_outlier_file"]),
                "assignment": r["assignment"],
                # Empty fields the labeller fills in:
                "label_extraction_worthy": None,  # yes / no / uncertain / flagged-spec
                "label_mechanism": None,          # background / reusable_scenario / shared_higher_level_step / unsure / n/a
                "labeller_notes": "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {len(pool_df)} entries written")

    # Stats
    stats = {
        "n_total": int(len(pool_df)),
        "n_overlap": N_OVERLAP,
        "n_per_author": n_each,
        "by_stratum": pool_df["stratum"].value_counts().to_dict(),
        "by_L_bucket": pool_df["L_bucket"].value_counts().to_dict(),
        "by_scope": pool_df["scope"].value_counts().to_dict(),
        "by_support_bucket": pool_df["sup_bucket"].value_counts().to_dict(),
        "n_examples_on_outlier_file": int(pool_df["example_on_outlier_file"].sum()),
    }
    out_stats = METHODOLOGY_DIR / "labeling_pool_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out_stats}")

    print("\n=== pool stratification ===")
    print(f"  by L bucket:")
    for k, v in sorted(stats["by_L_bucket"].items()):
        print(f"    {k}: {v}")
    print(f"  by scope:")
    for k, v in sorted(stats["by_scope"].items()):
        print(f"    {k}: {v}")
    print(f"  by support bucket:")
    for k, v in sorted(stats["by_support_bucket"].items()):
        print(f"    {k}: {v}")
    print(f"  overlap: {stats['n_overlap']} (labelled by all 3)")
    print(f"  per-author: {stats['n_per_author']}")


if __name__ == "__main__":
    main()
