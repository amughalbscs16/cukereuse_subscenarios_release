"""Generate the second batch of result figures for paper-2.

New outputs (PDF, vector) under paper/figures/:
  fig_feature_importance.pdf     XGBoost feature importance, Phase 6
  fig_classifier_per_fold.pdf    per-fold P/R/F1/AUC + bootstrap CI
  fig_top_rq3_patterns.pdf       top-20 cross-org RQ3 patterns by quality_score
  fig_quality_score_dist.pdf     distribution of quality_score per mechanism
"""
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------------------------------------------------------------
# (1) Feature importance
# ----------------------------------------------------------------------
with (ANA / "extraction_classifier_feature_importance.json").open() as f:
    feat = json.load(f)
feat_items = [(k, v) for k, v in feat.items() if v > 0]
feat_items.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(6.5, 3.0))
labels = [k for k, _ in feat_items]
vals   = [v for _, v in feat_items]
ax.barh(np.arange(len(labels)), vals,
        color="#5A8FBC", edgecolor="black", linewidth=0.4)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels([lbl.replace("_", "\\_") for lbl in labels])
ax.set_xlabel("XGBoost mean fold-level feature importance")
ax.set_title("Phase-6 classifier feature importance (extraction-worthy gate)")
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(True, axis="x", linestyle=":", linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_feature_importance.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

# ----------------------------------------------------------------------
# (2) Per-fold P/R/F1/AUC + bootstrap CI
# ----------------------------------------------------------------------
with (ANA / "extraction_classifier_metrics.json").open() as f:
    cls = json.load(f)
folds = cls["fold_metrics"]
metrics = ["precision", "recall", "f1", "auc"]
nice = {"precision":"Precision", "recall":"Recall", "f1":"$F_1$",
        "auc":"ROC-AUC"}

fig, ax = plt.subplots(figsize=(6.5, 2.8))
xs = np.arange(len(metrics))
w  = 0.13
for i, fld in enumerate(folds):
    ys = [fld[m] for m in metrics]
    ax.bar(xs + (i - 2)*w, ys, width=w, alpha=0.55,
           label=f"fold {fld['fold']}", edgecolor="black", linewidth=0.3)

# overlay bootstrap median + CI
ci_y = [cls["metrics_with_95ci"][m]["median"] for m in metrics]
ci_lo = [cls["metrics_with_95ci"][m]["ci_lo"] for m in metrics]
ci_hi = [cls["metrics_with_95ci"][m]["ci_hi"] for m in metrics]
ax.errorbar(xs, ci_y,
            yerr=[np.array(ci_y) - np.array(ci_lo),
                  np.array(ci_hi) - np.array(ci_y)],
            fmt="D", color="black", markersize=5, ecolor="black",
            elinewidth=1.0, capsize=3, label="bootstrap median + 95\\,\\% CI")
ax.set_xticks(xs)
ax.set_xticklabels([nice[m] for m in metrics])
ax.set_ylim(0.7, 1.02)
ax.set_ylabel("score")
ax.set_title("Phase-6 classifier: per-fold scores and bootstrap 95\\,\\% CIs")
ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=7.5)
ax.grid(True, axis="y", linestyle=":", linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_classifier_per_fold.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

# ----------------------------------------------------------------------
# (3) Top-20 RQ3 patterns (horizontal) and quality score distribution
# ----------------------------------------------------------------------
print("loading filtered candidates CSV...")
df = pd.read_csv(ANA / "extraction_candidates_filtered.csv",
                 usecols=["pattern", "L", "scope", "mechanism",
                          "support_total", "n_distinct_scenarios",
                          "n_distinct_orgs", "quality_score"])
print(f"  loaded {len(df):,} rows")

rq3 = df[df["scope"] == "RQ3"].sort_values(
        "quality_score", ascending=False).head(20).copy()
rq3["short"] = (rq3["pattern"].str.split(",").str[:6].str.join(",")
                + " ...")
rq3["short"] = rq3.apply(
    lambda r: f"L={int(r['L'])}, $\\nu_{{org}}$={int(r['n_distinct_orgs'])} | "
              f"{r['short'][:55]}",
    axis=1)
rq3 = rq3.iloc[::-1]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(np.arange(len(rq3)), rq3["quality_score"],
        color="#C26B43", edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(rq3)))
ax.set_yticklabels(rq3["short"], fontsize=7.5)
ax.set_xlabel("quality\\_score = "
              "$n_{\\mathrm{scen}} \\cdot \\sqrt{\\mathrm{support}}"
              "\\cdot n_{\\mathrm{orgs}}$")
ax.set_title("Top-20 cross-organisational (RQ3) patterns by quality score")
ax.grid(True, axis="x", linestyle=":", linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_top_rq3_patterns.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

# ----------------------------------------------------------------------
# (4) quality_score distribution per mechanism (boxen / violin)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.8))
order = ["background", "reusable_scenario", "shared_higher_level_step"]
pos = np.arange(len(order))
data = [df.loc[df["mechanism"] == m, "quality_score"].values for m in order]
parts = ax.violinplot(data, positions=pos, widths=0.7,
                      showmeans=False, showmedians=True,
                      showextrema=False)
colors = ["#5A8FBC", "#9DBF94", "#E08030"]
for pc, c in zip(parts["bodies"], colors):
    pc.set_facecolor(c); pc.set_edgecolor("black"); pc.set_alpha(0.55)
ax.set_yscale("log")
ax.set_xticks(pos)
ax.set_xticklabels([m.replace("_", "\\_") for m in order])
ax.set_ylabel("quality\\_score (log scale)")
ax.set_title("Distribution of quality\\_score across mechanisms (post-R6, "
             f"$n={len(df):,}$ surviving patterns)")
ax.grid(True, axis="y", which="both", linestyle=":",
        linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_quality_score_dist.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

print("done.")
