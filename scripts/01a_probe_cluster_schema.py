"""Probe the schemas of clusters_hybrid + cluster_members_hybrid.

Phase 1 needs to map every step row in steps.parquet to its cukereuse
hybrid cluster_id. This script figures out how the join works.
"""

from __future__ import annotations

from _parquet_loader import (
    load_hybrid_clusters,
    load_hybrid_members,
    load_steps,
)

print("=== clusters_hybrid.parquet ===")
clusters = load_hybrid_clusters()
print(f"rows: {len(clusters):,}")
print("schema:", list(clusters.columns))
print(clusters.head(3).to_string())
print()

print("=== cluster_members_hybrid.parquet ===")
members = load_hybrid_members()
print(f"rows: {len(members):,}")
print("schema:", list(members.columns))
print(members.head(3).to_string())
print()
print("dtypes:")
print(members.dtypes)
print()

print("=== steps.parquet (key columns only) ===")
steps = load_steps(columns=["repo_slug", "file_path", "line", "text"])
print(f"rows: {len(steps):,}")
print("schema:", list(steps.columns))
print(steps.head(3).to_string())
print()

# Try to figure out the join key
common_cols = set(steps.columns) & set(members.columns)
print(f"common columns between steps and members: {common_cols}")
