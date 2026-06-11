"""Phase 12 — decision-threshold sensitivity of the post-EW headline,
plus imbalance-appropriate OOF metrics vs the trivial all-yes baseline.

Replicates the Phase-9b rollup at classifier probability cutoffs
0.3/0.4/0.5/0.6/0.7 (0.5 must reproduce post_classifier_headline.json),
and derives MCC / balanced accuracy / negative-class P-R for the
out-of-fold confusion matrix implied exactly by the released OOF
metrics (P = 131/151, R = 131/143 -> TP=131 FP=20 FN=12 TN=34),
together with McNemar vs the all-yes predictor.

Inputs:
  analysis/slices.parquet
  analysis/extraction_classifier_predictions.parquet

Output:
  analysis/threshold_sensitivity.json
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def chi2_sf_df1(x: float) -> float:
    """Survival function of chi-squared with 1 df."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi-squared and p (matches paper's convention)."""
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return chi2, chi2_sf_df1(chi2)


def oof_baseline_metrics() -> dict:
    # Exact OOF confusion matrix implied by released metrics:
    # precision 0.86755 = 131/151, recall 0.91608 = 131/143 (integer-consistent).
    tp, fp, fn, tn = 131, 20, 12, 34
    n_pos, n_neg = tp + fn, tn + fp
    assert (n_pos, n_neg) == (143, 54)

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    xgb = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_pos": tp / (tp + fp),
        "recall_pos": tp / n_pos,
        "f1_pos": f1(tp / (tp + fp), tp / n_pos),
        "precision_neg": tn / (tn + fn),
        "recall_neg": tn / n_neg,
        "f1_neg": f1(tn / (tn + fn), tn / n_neg),
        "balanced_accuracy": 0.5 * (tp / n_pos + tn / n_neg),
        "mcc": mcc_num / mcc_den,
    }
    # All-yes: predicts positive everywhere.
    allyes = {
        "tp": n_pos, "fp": n_neg, "fn": 0, "tn": 0,
        "precision_pos": n_pos / (n_pos + n_neg),
        "recall_pos": 1.0,
        "f1_pos": f1(n_pos / (n_pos + n_neg), 1.0),
        "precision_neg": 0.0, "recall_neg": 0.0, "f1_neg": 0.0,
        "balanced_accuracy": 0.5,
        "mcc": 0.0,
    }
    # McNemar XGBoost vs all-yes: discordants are
    #   b = XGB right & all-yes wrong = TN (negatives XGB catches) = 34
    #   c = all-yes right & XGB wrong = FN (positives XGB misses) = 12
    chi2, p = mcnemar(34, 12)
    return {"xgboost_oof": xgb, "all_yes": allyes,
            "mcnemar_xgb_vs_allyes": {"b_xgb_only_right": 34,
                                      "c_allyes_only_right": 12,
                                      "chi2_cc": chi2, "p": p}}


def main() -> None:
    t0 = time.time()
    preds = pd.read_parquet(ANALYSIS / "extraction_classifier_predictions.parquet")
    print(f"{len(preds):,} scope-eligible patterns")

    slices = pd.read_parquet(
        ANALYSIS / "slices.parquet",
        columns=["repo_slug", "file_path", "scenario", "cluster_id_seq"],
    )
    slices["pattern"] = [",".join(map(str, s)) for s in slices["cluster_id_seq"]]
    print(f"{len(slices):,} slices")

    scen_key = ["repo_slug", "file_path", "scenario"]
    n_scenarios = slices.groupby(scen_key, sort=False).ngroups
    n_repos = slices["repo_slug"].nunique()
    print(f"{n_scenarios:,} scenarios, {n_repos} repos")

    sweep = {}
    for t in THRESHOLDS:
        ew = preds[preds["p_extraction_worthy"] >= t]
        rq1 = set(ew.loc[ew["max_within_file_recurrence"] >= 2, "pattern"])
        rq2 = set(ew.loc[ew["max_within_repo_files"] >= 2, "pattern"])
        rq3 = set(ew.loc[ew["n_distinct_orgs"] >= 2, "pattern"])

        flags = pd.DataFrame({
            "rq1": slices["pattern"].isin(rq1),
            "rq2": slices["pattern"].isin(rq2),
            "rq3": slices["pattern"].isin(rq3),
        })
        flags[scen_key] = slices[scen_key]
        per_scen = flags.groupby(scen_key, sort=False)[["rq1", "rq2", "rq3"]].any()
        per_repo = flags.groupby("repo_slug", sort=False)[["rq2", "rq3"]].any()

        sweep[str(t)] = {
            "n_patterns_ew": int(len(ew)),
            "pct_patterns_ew": round(100 * len(ew) / len(preds), 1),
            "scenario_pct_rq1": round(100 * per_scen["rq1"].mean(), 2),
            "scenario_pct_rq2": round(100 * per_scen["rq2"].mean(), 2),
            "scenario_pct_rq3": round(100 * per_scen["rq3"].mean(), 2),
            "repo_pct_rq2": round(100 * per_repo["rq2"].mean(), 2),
            "repo_pct_rq3": round(100 * per_repo["rq3"].mean(), 2),
        }
        print(f"t={t}: {sweep[str(t)]}")

    out = {
        "thresholds": sweep,
        "baseline_comparison": oof_baseline_metrics(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    path = ANALYSIS / "threshold_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
