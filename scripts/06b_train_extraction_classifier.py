"""Phase 6 — train the binary extraction-worthy classifier.

Inputs:
  methodology/labels.jsonl                          (200 labelled slices)
  analysis/exact_subsequence_ranking.parquet        (full pattern feature set)

Target:
  Binary label_extraction_worthy:
    positive (1) = "yes"
    negative (0) = "no" | "uncertain" | "flagged-spec"
  "tie" entries (3 records) are excluded from training.

Features:
  L, support_total, n_distinct_repos, n_distinct_orgs, n_distinct_files,
  max_within_file_recurrence, max_within_repo_files, outlier_fraction,
  has_template_structure (bool), scope_one_hot (RQ1/RQ2/RQ3),
  ratio_within_file = max_within_file_recurrence / support_total,
  ratio_within_repo = max_within_repo_files / support_total,
  ratio_orgs        = n_distinct_orgs / max(n_distinct_repos, 1).

Method:
  Stratified 5-fold cross-validation; XGBoost binary classifier;
  bootstrap 95% CIs on precision, recall, F1 over fold predictions.

Outputs:
  analysis/extraction_classifier_metrics.json     (precision/recall/F1 with bootstrap CIs)
  analysis/extraction_classifier_predictions.parquet  (full ranking with
      predicted P(extraction-worthy))
  analysis/extraction_classifier_feature_importance.json
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "methodology" / "labels.jsonl"
RANK = ROOT / "analysis" / "exact_subsequence_ranking.parquet"
OUT_METRICS = ROOT / "analysis" / "extraction_classifier_metrics.json"
OUT_PREDS = ROOT / "analysis" / "extraction_classifier_predictions.parquet"
OUT_FI = ROOT / "analysis" / "extraction_classifier_feature_importance.json"

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000


def main() -> None:
    t0 = time.time()
    labels = [json.loads(l) for l in LABELS.open(encoding="utf-8")]
    print(f"loaded labels: {len(labels):,}")

    rank = pd.read_parquet(RANK)
    print(f"loaded ranking: {len(rank):,} patterns")

    # -------- build feature matrix on labelled set --------
    df = pd.DataFrame(labels)
    df = df.merge(
        rank[["pattern", "n_distinct_orgs", "has_template_structure"]],
        on="pattern", how="left"
    )
    # tied/missing entries dropped
    df = df[df["label_extraction_worthy"].isin(["yes", "no", "uncertain", "flagged-spec"])].copy()
    df["y"] = (df["label_extraction_worthy"] == "yes").astype(int)
    print(f"training set: {len(df):,}  (positives: {df['y'].sum()},  negatives: {(1-df['y']).sum()})")

    df["scope_RQ1"] = (df["scope"] == "RQ1").astype(int)
    df["scope_RQ2"] = (df["scope"] == "RQ2").astype(int)
    df["scope_RQ3"] = (df["scope"] == "RQ3").astype(int)
    df["has_template_structure"] = df["has_template_structure"].astype(int)
    df["ratio_within_file"] = df["max_within_file_recurrence"] / df["support_total"].clip(lower=1)
    df["ratio_within_repo"] = df["max_within_repo_files"] / df["support_total"].clip(lower=1)
    df["ratio_orgs"] = df["n_distinct_orgs"] / df["n_distinct_repos"].clip(lower=1)

    feat_cols = [
        "L", "support_total", "n_distinct_repos", "n_distinct_orgs",
        "n_distinct_files", "max_within_file_recurrence",
        "max_within_repo_files", "outlier_fraction",
        "has_template_structure", "scope_RQ1", "scope_RQ2", "scope_RQ3",
        "ratio_within_file", "ratio_within_repo", "ratio_orgs",
    ]
    X = df[feat_cols].values.astype(float)
    y = df["y"].values

    # -------- 5-fold CV with bootstrap CIs --------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(df))
    oof_prob = np.zeros(len(df))
    fold_metrics = []
    fis = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=4,
        )
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_pred[te] = pred
        oof_prob[te] = prob
        fis.append(clf.feature_importances_)
        m = {
            "fold": fold,
            "precision": float(precision_score(y[te], pred, zero_division=0)),
            "recall":    float(recall_score(y[te], pred, zero_division=0)),
            "f1":        float(f1_score(y[te], pred, zero_division=0)),
            "auc":       float(roc_auc_score(y[te], prob)) if len(np.unique(y[te]))>1 else float("nan"),
            "n_test":    int(len(te)),
        }
        fold_metrics.append(m)
        print(f"  fold {fold}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} AUC={m['auc']:.3f}")

    # -------- bootstrap CIs on OOF predictions --------
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try:
            boots.append({
                "precision": precision_score(y[idx], oof_pred[idx], zero_division=0),
                "recall":    recall_score(y[idx], oof_pred[idx], zero_division=0),
                "f1":        f1_score(y[idx], oof_pred[idx], zero_division=0),
                "auc":       roc_auc_score(y[idx], oof_prob[idx]) if len(np.unique(y[idx]))>1 else float("nan"),
            })
        except Exception:
            continue

    def ci(name: str) -> tuple[float, float, float]:
        vals = np.array([b[name] for b in boots if not np.isnan(b[name])])
        return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    summary_metrics = {
        "n_train": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
        "fold_metrics": fold_metrics,
        "bootstrap_n": N_BOOTSTRAP,
        "metrics_with_95ci": {
            name: {"median": m, "ci_lo": lo, "ci_hi": hi}
            for name, (m, lo, hi) in [(k, ci(k)) for k in ("precision","recall","f1","auc")]
        },
        "oof": {
            "precision": float(precision_score(y, oof_pred, zero_division=0)),
            "recall":    float(recall_score(y, oof_pred, zero_division=0)),
            "f1":        float(f1_score(y, oof_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y, oof_prob)),
        },
    }
    OUT_METRICS.write_text(json.dumps(summary_metrics, indent=2))
    print(f"\noverall OOF: P={summary_metrics['oof']['precision']:.3f} "
          f"R={summary_metrics['oof']['recall']:.3f} "
          f"F1={summary_metrics['oof']['f1']:.3f} "
          f"AUC={summary_metrics['oof']['auc']:.3f}")
    print(f"95% CI on F1: {summary_metrics['metrics_with_95ci']['f1']}")

    # -------- feature importance --------
    fi_mean = np.mean(fis, axis=0).tolist()
    fi_dict = dict(sorted(zip(feat_cols, fi_mean), key=lambda kv: -kv[1]))
    OUT_FI.write_text(json.dumps(fi_dict, indent=2))
    print(f"\nfeature importance (top 5):")
    for k, v in list(fi_dict.items())[:5]:
        print(f"  {k:32s} {v:.3f}")

    # -------- final fit on all training data; predict on full ranking --------
    print("\nfitting final model on all 200 labels and predicting on full ranking ...")
    clf_full = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )
    clf_full.fit(X, y)

    # Build feature matrix for the full ranking (619,827 patterns with at least one scope signal)
    # We need to compute outlier_fraction per pattern; reuse the labels' notion.
    # The simpler approach: since outlier_fraction is in labels.jsonl but not in ranking,
    # recompute it from slices.parquet.
    from time import time as _t
    t1 = _t()
    print("  computing outlier_fraction per pattern ...")
    slices = pd.read_parquet(
        ROOT / "analysis" / "slices.parquet",
        columns=["repo_slug", "file_path", "cluster_id_seq"],
    )
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]
    out = pd.read_csv(ROOT / "analysis" / "spec_suite_outliers.csv")
    keys = set(out["repo_slug"] + "|" + out["file_path"])
    slices["on_outlier"] = (slices["repo_slug"]+"|"+slices["file_path"]).isin(keys)
    of = slices.groupby("pattern", sort=False)["on_outlier"].mean().rename("outlier_fraction").reset_index()
    rank2 = rank.merge(of, on="pattern", how="left")
    rank2["outlier_fraction"] = rank2["outlier_fraction"].fillna(0.0)
    print(f"  done ({_t()-t1:.1f}s)")

    # Restrict to patterns with at least one scope signal
    rq1 = rank2["max_within_file_recurrence"] >= 2
    rq2 = rank2["max_within_repo_files"] >= 2
    rq3 = rank2["n_distinct_orgs"] >= 2
    rank2["scope"] = "noscope"
    rank2.loc[rq1, "scope"] = "RQ1"
    rank2.loc[rq2, "scope"] = "RQ2"   # later overwrites RQ1 — most-specific
    rank2.loc[rq3, "scope"] = "RQ3"   # later overwrites — most-specific
    rank2 = rank2[rank2["scope"] != "noscope"].copy()
    print(f"  scope-eligible patterns: {len(rank2):,}")

    rank2["scope_RQ1"] = (rank2["scope"] == "RQ1").astype(int)
    rank2["scope_RQ2"] = (rank2["scope"] == "RQ2").astype(int)
    rank2["scope_RQ3"] = (rank2["scope"] == "RQ3").astype(int)
    rank2["has_template_structure"] = rank2["has_template_structure"].astype(int)
    rank2["ratio_within_file"] = rank2["max_within_file_recurrence"] / rank2["support_total"].clip(lower=1)
    rank2["ratio_within_repo"] = rank2["max_within_repo_files"] / rank2["support_total"].clip(lower=1)
    rank2["ratio_orgs"] = rank2["n_distinct_orgs"] / rank2["n_distinct_repos"].clip(lower=1)

    Xfull = rank2[feat_cols].values.astype(float)
    print(f"  predicting on {len(Xfull):,} patterns ...")
    rank2["p_extraction_worthy"] = clf_full.predict_proba(Xfull)[:, 1]
    rank2["pred_extraction_worthy"] = (rank2["p_extraction_worthy"] >= 0.5).astype(int)

    out_cols = ["pattern", "L", "scope", "support_total", "n_distinct_repos",
                "n_distinct_orgs", "n_distinct_files",
                "max_within_file_recurrence", "max_within_repo_files",
                "outlier_fraction", "has_template_structure",
                "p_extraction_worthy", "pred_extraction_worthy"]
    rank2[out_cols].to_parquet(OUT_PREDS, compression="zstd", index=False)
    print(f"  wrote {OUT_PREDS}")

    n_pred_pos = int(rank2["pred_extraction_worthy"].sum())
    print(f"\n  patterns predicted extraction-worthy: {n_pred_pos:,} of {len(rank2):,} "
          f"({100*n_pred_pos/len(rank2):.1f}%)")

    print(f"\ntotal wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
