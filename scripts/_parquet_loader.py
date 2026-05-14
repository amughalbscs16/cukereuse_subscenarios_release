"""Lightweight loaders for the cukereuse-release parquet artefacts.

Used by the subscenario-mining scripts so they don't need to re-implement
the repo-traversal logic each time. Reads from a local copy in
``corpus/`` if present; otherwise falls back to a sibling
``cukereuse-release/corpus`` directory (the upstream cukereuse repo
checked out next to this one). The fallback path can be overridden via
the ``CUKEREUSE_RELEASE_CORPUS`` environment variable.
"""

from __future__ import annotations

import os
import pathlib

import pandas as pd

LOCAL_CORPUS = pathlib.Path(__file__).resolve().parent.parent / "corpus"
RELEASE_CORPUS = pathlib.Path(
    os.environ.get(
        "CUKEREUSE_RELEASE_CORPUS",
        str(pathlib.Path(__file__).resolve().parent.parent.parent
            / "cukereuse-release" / "corpus"),
    )
)


def _resolve(name: str) -> pathlib.Path:
    local = LOCAL_CORPUS / name
    if local.exists():
        return local
    release = RELEASE_CORPUS / name
    if release.exists():
        return release
    raise FileNotFoundError(
        f"Could not find {name} in {LOCAL_CORPUS} or {RELEASE_CORPUS}. "
        f"Either copy the parquet into corpus/ or pull the cukereuse-release repo."
    )


def load_steps(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the 1.1M-row step table."""
    return pd.read_parquet(_resolve("steps.parquet"), columns=columns)


def load_hybrid_clusters() -> pd.DataFrame:
    """Load the 65,242-cluster hybrid summary table (cluster_id, count, ...)."""
    return pd.read_parquet(_resolve("clusters_hybrid.parquet"))


def load_hybrid_members(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cluster-member table for the hybrid strategy."""
    return pd.read_parquet(_resolve("cluster_members_hybrid.parquet"), columns=columns)


def load_exact_clusters() -> pd.DataFrame:
    return pd.read_parquet(_resolve("clusters_exact.parquet"))


def load_exact_members(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(_resolve("cluster_members_exact.parquet"), columns=columns)


if __name__ == "__main__":
    s = load_steps(columns=["repo", "scenario", "keyword", "text"])
    print(f"steps: {len(s):,} rows, {s['repo'].nunique()} repos, "
          f"{s.groupby(['repo','scenario']).ngroups:,} scenarios")
