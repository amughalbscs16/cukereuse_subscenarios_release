"""Phase 8 — mechanism classifier.

Predicts the Mughal-2024 mechanism for each extraction-worthy slice:
  - background
  - reusable_scenario
  - shared_higher_level_step
  - unsure (none in current label set)

Training set: the 143 labels with label_extraction_worthy = "yes" from
the three-author labelling.

Two predictors compared:
  (a) Rule-based scope-driven baseline:
      RQ1 -> background, RQ2 -> reusable_scenario, RQ3 -> shared_higher_level_step
  (b) Learned XGBoost multi-class classifier on the same features as Phase 6.

Metric: macro-F1 with bootstrap 95% CIs and per-class precision/recall.

Output: analysis/mechanism_classifier_metrics.json
        analysis/mechanism_predictions.parquet (final mechanism per
            predicted-extraction-worthy pattern)
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "methodology" / "labels.jsonl"
RANK = ROOT / "analysis" / "exact_subsequence_ranking.parquet"
PREDS_EW = ROOT / "analysis" / "extraction_classifier_predictions.parquet"
OUT_METRICS = ROOT / "analysis" / "mechanism_classifier_metrics.json"
OUT_PREDS = ROOT / "analysis" / "mechanism_predictions.parquet"

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000

MECH_CLASSES = ["background", "reusable_scenario", "shared_higher_level_step"]


def main() -> None:
    t0 = time.time()
    labels = [json.loads(l) for l in LABELS.open(encoding="utf-8")]
    rank = pd.read_parquet(RANK)
    df = pd.DataFrame(labels)
    df = df.merge(rank[["pattern", "n_distinct_orgs", "has_template_structure"]],
                  on="pattern", how="left")
    df = df[df["label_extraction_worthy"] == "yes"].copy()
    df = df[df["label_mechanism"].isin(MECH_CLASSES)].copy()
    print(f"training set (yes-only, mechanism in known classes): {len(df):,}")
    print("scope x mechanism:")
    print(df.groupby(["scope", "label_mechanism"]).size())

    # ----- rule-based baseline -----
    rule_pred = df["scope"].map({
        "RQ1": "background",
        "RQ2": "reusable_scenario",
        "RQ3": "shared_higher_level_step",
    })
    rule_acc = (rule_pred == df["label_mechanism"]).mean()
    rule_f1 = f1_score(df["label_mechanism"], rule_pred, labels=MECH_CLASSES, average="macro")
    print(f"\n=== rule-based scope-driven baseline ===")
    print(f"  accuracy: {rule_acc:.3f}")
    print(f"  macro F1: {rule_f1:.3f}")
    print(classification_report(df["label_mechanism"], rule_pred, labels=MECH_CLASSES))

    # ----- XGBoost multi-class -----
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
    cls_to_idx = {c: i for i, c in enumerate(MECH_CLASSES)}
    X = df[feat_cols].values.astype(float)
    y = np.array([cls_to_idx[c] for c in df["label_mechanism"]])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(df), dtype=int)
    fold_metrics = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            objective="multi:softprob", num_class=len(MECH_CLASSES),
            random_state=SEED, n_jobs=4,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        oof_pred[te] = pred
        m = {
            "fold": fold,
            "accuracy": float((pred == y[te]).mean()),
            "macro_f1": float(f1_score(y[te], pred, labels=list(range(len(MECH_CLASSES))), average="macro")),
            "n_test": int(len(te)),
        }
        fold_metrics.append(m)
        print(f"  fold {fold}: acc={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f}")

    # bootstrap CIs on OOF predictions
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y), size=len(y), replace=True)
        boots.append({
            "accuracy": float((oof_pred[idx] == y[idx]).mean()),
            "macro_f1": float(f1_score(y[idx], oof_pred[idx],
                labels=list(range(len(MECH_CLASSES))), average="macro", zero_division=0)),
        })

    def ci(name):
        v = np.array([b[name] for b in boots])
        return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    xgb_oof_acc = float((oof_pred == y).mean())
    xgb_oof_f1 = float(f1_score(y, oof_pred, labels=list(range(len(MECH_CLASSES))), average="macro"))
    pred_names = [MECH_CLASSES[i] for i in oof_pred]
    print(f"\n=== XGBoost mechanism classifier (5-fold CV) ===")
    print(f"  OOF accuracy: {xgb_oof_acc:.3f}")
    print(f"  OOF macro F1: {xgb_oof_f1:.3f}")
    print(classification_report(df["label_mechanism"], pred_names, labels=MECH_CLASSES))

    # ----- final fit & predict on extraction-worthy population -----
    print("\nfitting final mechanism model on all 143 yes labels and predicting "
          "on the predicted-extraction-worthy population ...")
    clf_full = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=len(MECH_CLASSES),
        random_state=SEED, n_jobs=4,
    )
    clf_full.fit(X, y)

    ew = pd.read_parquet(PREDS_EW)
    ew = ew[ew["pred_extraction_worthy"] == 1].copy()
    print(f"  applying to {len(ew):,} predicted-EW patterns")
    ew["scope_RQ1"] = (ew["scope"] == "RQ1").astype(int)
    ew["scope_RQ2"] = (ew["scope"] == "RQ2").astype(int)
    ew["scope_RQ3"] = (ew["scope"] == "RQ3").astype(int)
    ew["has_template_structure"] = ew["has_template_structure"].astype(int)
    ew["ratio_within_file"] = ew["max_within_file_recurrence"] / ew["support_total"].clip(lower=1)
    ew["ratio_within_repo"] = ew["max_within_repo_files"] / ew["support_total"].clip(lower=1)
    ew["ratio_orgs"] = ew["n_distinct_orgs"] / ew["n_distinct_repos"].clip(lower=1)
    Xfull = ew[feat_cols].values.astype(float)
    pred_idx = clf_full.predict(Xfull)
    ew["mechanism"] = [MECH_CLASSES[i] for i in pred_idx]

    # Also rule-based prediction for comparison
    ew["mechanism_rule"] = ew["scope"].map({
        "RQ1": "background", "RQ2": "reusable_scenario", "RQ3": "shared_higher_level_step",
    })
    print(f"\n  final mechanism distribution (XGBoost):")
    for k, v in ew["mechanism"].value_counts().items():
        print(f"    {k:35s} {v:8,}")
    print(f"  final mechanism distribution (rule-based, for comparison):")
    for k, v in ew["mechanism_rule"].value_counts().items():
        print(f"    {k:35s} {v:8,}")
    agreement = (ew["mechanism"] == ew["mechanism_rule"]).mean()
    print(f"  rule-vs-XGBoost agreement on EW patterns: {agreement:.3f}")

    out_cols = ["pattern", "L", "scope", "support_total",
                "n_distinct_orgs", "n_distinct_repos",
                "max_within_file_recurrence", "max_within_repo_files",
                "outlier_fraction", "p_extraction_worthy",
                "mechanism", "mechanism_rule"]
    ew[out_cols].to_parquet(OUT_PREDS, compression="zstd", index=False)
    print(f"  wrote {OUT_PREDS}")

    # ----- summary -----
    summary = {
        "n_train_yes_only": int(len(df)),
        "label_distribution": df["label_mechanism"].value_counts().to_dict(),
        "rule_based_baseline": {
            "accuracy": float(rule_acc),
            "macro_f1": float(rule_f1),
            "interpretation": "RQ1->background, RQ2->reusable_scenario, "
                              "RQ3->shared_higher_level_step",
        },
        "xgb_classifier": {
            "fold_metrics": fold_metrics,
            "oof_accuracy": xgb_oof_acc,
            "oof_macro_f1": xgb_oof_f1,
            "metrics_with_95ci": {
                k: {"median": m, "ci_lo": lo, "ci_hi": hi}
                for k, (m, lo, hi) in [(k, ci(k)) for k in ("accuracy", "macro_f1")]
            },
        },
        "applied_to_ew_population": {
            "n_predicted_ew": int(len(ew)),
            "mechanism_xgb": ew["mechanism"].value_counts().to_dict(),
            "mechanism_rule": ew["mechanism_rule"].value_counts().to_dict(),
            "rule_vs_xgb_agreement": float(agreement),
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT_METRICS.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_METRICS}")
    print(f"total wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
