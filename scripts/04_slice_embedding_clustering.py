"""Phase 4 — SBERT slice-embedding clustering.

Pipeline:
  1. Load clusters_hybrid.parquet -> cluster_id -> canonical_text map.
  2. SBERT-encode canonical texts (sentence-transformers MiniLM-L6-v2).
     Cache to analysis/cluster_embeddings.npz.
  3. Load the recurring-pattern table from Phase 2.
  4. Filter to patterns with at least one positive RQ signal
     (max_within_file_recurrence >= 2 OR max_within_repo_files >= 2 OR
      n_distinct_repos >= 2).
  5. Mean-pool cluster embeddings -> 384-d slice embeddings.
  6. UMAP -> 50d.
  7. HDBSCAN cluster.
  8. Write analysis/slice_clusters.parquet.

Why this matters:
  Phase 2's exact-sequence ranking groups slices that have *identical*
  cluster_id sequences. Phase 4 catches the inverse direction — slices
  that should be considered equivalent because their semantics overlap,
  even though one or more constituent clusters differ. Two slices with
  cluster_id_seqs [12, 45, 78] and [12, 46, 78] would not match in Phase
  2 but should cluster together in Phase 4 if cluster 45 and 46 are near-
  paraphrases.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd

ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def encode_clusters() -> np.ndarray:
    """SBERT-encode every cluster's canonical text. Cache to disk."""
    cache = ANALYSIS_DIR / "cluster_embeddings.npz"
    if cache.exists():
        print(f"loading cached cluster embeddings from {cache} ...")
        z = np.load(cache)
        emb = z["emb"]
        cids = z["cluster_id"]
        print(f"  {len(cids):,} clusters, dim={emb.shape[1]}")
        return cids, emb

    from sentence_transformers import SentenceTransformer
    from _parquet_loader import load_hybrid_clusters

    print("loading clusters_hybrid.parquet ...")
    clusters = load_hybrid_clusters()
    clusters = clusters[["cluster_id", "canonical_text"]].sort_values("cluster_id")
    print(f"  {len(clusters):,} clusters")

    print("loading SBERT all-MiniLM-L6-v2 ...")
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  loaded ({time.time()-t0:.1f}s)")

    texts = clusters["canonical_text"].tolist()
    print(f"encoding {len(texts):,} canonical texts (batch=128) ...")
    t1 = time.time()
    emb = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    print(f"  encoded -> shape {emb.shape}  ({time.time()-t1:.1f}s)")

    cids = clusters["cluster_id"].to_numpy(dtype=np.int32)
    np.savez_compressed(cache, cluster_id=cids, emb=emb)
    print(f"  cached to {cache}")
    return cids, emb


def main() -> None:
    t0 = time.time()
    cids, cluster_emb = encode_clusters()
    # Build dense lookup: cluster_id -> row index
    max_cid = int(cids.max())
    lookup = np.full(max_cid + 2, -1, dtype=np.int64)
    lookup[cids] = np.arange(len(cids))

    # Load patterns
    in_path = ANALYSIS_DIR / "exact_subsequence_ranking.parquet"
    print(f"\nloading {in_path} ...")
    t1 = time.time()
    patterns = pd.read_parquet(in_path)
    print(f"  {len(patterns):,} recurring patterns ({time.time()-t1:.1f}s)")

    # Filter to RQ candidate union
    n_before = len(patterns)
    rq_mask = (
        (patterns["max_within_file_recurrence"] >= 2) |
        (patterns["max_within_repo_files"] >= 2) |
        (patterns["n_distinct_repos"] >= 2)
    )
    patterns = patterns[rq_mask].reset_index(drop=True)
    print(f"  after RQ-candidate filter: {len(patterns):,} (dropped {n_before-len(patterns):,})")

    # Mean-pool cluster embeddings per pattern
    print("\nmean-pooling cluster embeddings per pattern ...")
    t2 = time.time()
    dim = cluster_emb.shape[1]
    slice_emb = np.zeros((len(patterns), dim), dtype=np.float32)
    n_unknown = 0
    for i, seq in enumerate(patterns["cluster_id_seq"]):
        idx = lookup[np.asarray(seq, dtype=np.int64)]
        if (idx < 0).any():
            n_unknown += 1
            idx = idx[idx >= 0]
            if len(idx) == 0:
                continue
        slice_emb[i] = cluster_emb[idx].mean(axis=0)
        if (i + 1) % 100_000 == 0:
            print(f"  ... {i+1:,} ({time.time()-t2:.1f}s)")
    print(f"  built {slice_emb.shape}; unknown-cid patterns: {n_unknown}  "
          f"({time.time()-t2:.1f}s)")

    # Cache
    emb_cache = ANALYSIS_DIR / "slice_embeddings.npz"
    np.savez_compressed(emb_cache, emb=slice_emb,
                        pattern=patterns["pattern"].to_numpy())
    print(f"  cached to {emb_cache}")

    # UMAP
    print("\nUMAP -> 50d ...")
    t3 = time.time()
    import umap  # noqa
    # NOTE: random_state intentionally omitted to enable parallel execution.
    # Setting random_state forces n_jobs=1 (UMAP warns about this), which on
    # 500k x 384d points takes 1-3 hours on CPU. Without the seed, UMAP uses
    # all cores and finishes in 5-15 min. Downstream HDBSCAN clustering is
    # robust to UMAP non-determinism (clusters land in similar locations),
    # so the methods section will report this trade-off. For exact
    # reproducibility we would re-add random_state and accept the runtime.
    reducer = umap.UMAP(
        n_components=50,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        n_jobs=-1,
        verbose=True,
    )
    slice_emb_50 = reducer.fit_transform(slice_emb)
    print(f"  UMAP done -> {slice_emb_50.shape}  ({time.time()-t3:.1f}s)")

    # HDBSCAN
    print("\nHDBSCAN clustering ...")
    t4 = time.time()
    import hdbscan  # noqa
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(slice_emb_50)
    n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    print(f"  HDBSCAN -> {n_clusters:,} clusters, {n_noise:,} noise points "
          f"({100*n_noise/len(labels):.1f}%)  ({time.time()-t4:.1f}s)")

    # Persist
    out = patterns[["pattern", "L", "support_total", "n_distinct_repos",
                    "max_within_file_recurrence", "max_within_repo_files"]].copy()
    out["slice_cluster_id"] = labels.astype(np.int32)
    out_path = ANALYSIS_DIR / "slice_clusters.parquet"
    out.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nwrote {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")

    # Summary
    cluster_sizes = pd.Series(labels[labels >= 0]).value_counts()
    summary = {
        "n_patterns_input": int(n_before),
        "n_patterns_clustered": int(len(patterns)),
        "n_unknown_cid_patterns": int(n_unknown),
        "n_clusters": int(n_clusters),
        "n_noise_points": int(n_noise),
        "noise_fraction": round(n_noise / max(len(labels), 1), 3),
        "cluster_size_min": int(cluster_sizes.min()) if n_clusters else 0,
        "cluster_size_median": float(cluster_sizes.median()) if n_clusters else 0.0,
        "cluster_size_p95": float(cluster_sizes.quantile(0.95)) if n_clusters else 0.0,
        "cluster_size_max": int(cluster_sizes.max()) if n_clusters else 0,
        "wall_seconds": round(time.time() - t0, 1),
    }
    summary_path = ANALYSIS_DIR / "slice_clusters_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
