"""Phase 1 — slice extraction with paraphrase-robust cluster-id sequences.

Locked decisions from Phase 0:
  - Scenario key:    (repo_slug, file_path, scenario)
  - Filter:          drop is_background=True, drop empty scenario, drop length<2
  - Step identity:   cukereuse hybrid cluster_id (from cluster_members_hybrid)
  - Slice lengths:   L in [2, L_MAX]; L_MAX = 18 (p95 of cleaned set)

Steps without a cluster_id (~7.4% of the corpus, 82,162 rows) are dropped
because their cluster identity is unknown — they cannot match any other
slice. We document the loss in the methods section.

Output: analysis/slices.parquet with columns
    slice_id, repo_slug, file_path, scenario, position_start, L,
    cluster_id_seq, text_seq

Each slice is one row. cluster_id_seq is a list of ints; text_seq is a
list of strings.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd

from _parquet_loader import load_hybrid_members, load_steps

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

L_MAX = 18


def main() -> None:
    t0 = time.time()
    print(f"loading steps.parquet ...")
    steps = load_steps(
        columns=["repo_slug", "file_path", "line", "keyword", "text",
                 "scenario", "is_background"]
    )
    print(f"  steps: {len(steps):,} rows ({time.time()-t0:.1f}s)")

    t1 = time.time()
    print("loading cluster_members_hybrid.parquet ...")
    members = load_hybrid_members(
        columns=["repo_slug", "file_path", "line", "cluster_id"]
    )
    print(f"  members: {len(members):,} rows ({time.time()-t1:.1f}s)")

    t2 = time.time()
    print("merging on (repo_slug, file_path, line) ...")
    df = steps.merge(
        members, on=["repo_slug", "file_path", "line"], how="left"
    )
    n_unclustered = int(df["cluster_id"].isna().sum())
    print(f"  merged: {len(df):,} rows; unclustered: {n_unclustered:,} "
          f"({100*n_unclustered/len(df):.2f}%)  ({time.time()-t2:.1f}s)")

    # Apply Key E filter
    t3 = time.time()
    print("applying Key E filter (drop is_background, empty-name, len<2, unclustered) ...")
    before = len(df)
    df = df[~df["is_background"]]
    df = df[df["scenario"].astype(str).str.strip() != ""]
    df = df[df["cluster_id"].notna()]
    df["cluster_id"] = df["cluster_id"].astype(np.int32)
    # Drop scenarios with fewer than 2 remaining steps
    scenario_lens = df.groupby(
        ["repo_slug", "file_path", "scenario"]
    )["line"].transform("size")
    df = df[scenario_lens >= 2]
    print(f"  rows after filter: {len(df):,} (dropped {before - len(df):,})  "
          f"({time.time()-t3:.1f}s)")

    # Sort within scenario by line
    t4 = time.time()
    print("sorting within scenario by line ...")
    df = df.sort_values(
        ["repo_slug", "file_path", "scenario", "line"], kind="stable"
    ).reset_index(drop=True)
    print(f"  sorted ({time.time()-t4:.1f}s)")

    # Slice generation — columnar accumulation for speed
    t5 = time.time()
    print(f"generating slices (L in [2, {L_MAX}]) ...")

    out_repo: list[str] = []
    out_file: list[str] = []
    out_scenario: list[str] = []
    out_position: list[int] = []
    out_L: list[int] = []
    out_cseq: list[list[int]] = []
    out_tseq: list[list[str]] = []

    n_scenarios_seen = 0
    n_scenarios_too_short = 0

    for (repo, fpath, scen), g in df.groupby(
        ["repo_slug", "file_path", "scenario"], sort=False
    ):
        n_scenarios_seen += 1
        cluster_ids = g["cluster_id"].to_numpy(dtype=np.int32)
        texts = g["text"].tolist()
        n = len(cluster_ids)
        if n < 2:
            n_scenarios_too_short += 1
            continue
        L_top = min(L_MAX, n)
        for L in range(2, L_top + 1):
            for start in range(0, n - L + 1):
                out_repo.append(repo)
                out_file.append(fpath)
                out_scenario.append(scen)
                out_position.append(start)
                out_L.append(L)
                out_cseq.append(cluster_ids[start:start + L].tolist())
                out_tseq.append(texts[start:start + L])

        if n_scenarios_seen % 20000 == 0:
            print(f"  ... {n_scenarios_seen:,} scenarios, "
                  f"{len(out_L):,} slices so far ({time.time()-t5:.1f}s)")

    print(f"  {n_scenarios_seen:,} scenarios processed; "
          f"{n_scenarios_too_short:,} too short")
    print(f"  total slices: {len(out_L):,}  ({time.time()-t5:.1f}s)")

    # Build dataframe
    t6 = time.time()
    print("building dataframe ...")
    slices_df = pd.DataFrame({
        "slice_id": np.arange(len(out_L), dtype=np.int64),
        "repo_slug": out_repo,
        "file_path": out_file,
        "scenario": out_scenario,
        "position_start": np.array(out_position, dtype=np.int32),
        "L": np.array(out_L, dtype=np.int8),
        "cluster_id_seq": out_cseq,
        "text_seq": out_tseq,
    })
    print(f"  built ({time.time()-t6:.1f}s); shape: {slices_df.shape}")

    # Write
    t7 = time.time()
    out_path = ANALYSIS_DIR / "slices.parquet"
    print(f"writing {out_path} ...")
    slices_df.to_parquet(out_path, compression="zstd", index=False)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {file_mb:.1f} MB  ({time.time()-t7:.1f}s)")

    # Summary
    summary = {
        "n_steps_total": int(len(steps)),
        "n_steps_unclustered": int(n_unclustered),
        "n_steps_after_filter": int(len(df)),
        "n_scenarios_after_filter": int(n_scenarios_seen),
        "L_max": L_MAX,
        "n_slices": int(len(out_L)),
        "slices_per_L": {
            int(L): int(c) for L, c in
            zip(*np.unique(np.array(out_L), return_counts=True))
        },
        "wall_seconds": round(time.time() - t0, 1),
        "output_mb": round(file_mb, 1),
    }
    summary_path = ANALYSIS_DIR / "slices_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
