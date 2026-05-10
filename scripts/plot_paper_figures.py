"""Generate three result figures for paper-2.

Outputs (all PDF, vector, embedded fonts) under paper/figures/:
  fig_L_distribution.pdf   pre/post R6 by slice length
  fig_mech_distribution.pdf  mechanism predictions on 464,073 EW
  fig_classifier_vs_llm.pdf  Phase-6 vs Phase-7 binary metrics
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Match elsarticle body width (~16cm)
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
# Figure 1: L-distribution pre vs post R1-R6
# ----------------------------------------------------------------------
print("loading extraction_candidates_index.csv (this is large)...")
df = pd.read_csv(
    ANA / "extraction_candidates_index.csv",
    usecols=["L", "flag_templated", "flag_repetition", "flag_single_scen",
             "flag_overlap_dom", "flag_not_cross_org", "flag_non_closed"],
)
print(f"  loaded {len(df):,} rows")
pre_counts = df["L"].value_counts().sort_index()
keep = (~df["flag_templated"]) & (~df["flag_repetition"]) & \
       (~df["flag_single_scen"]) & (~df["flag_overlap_dom"]) & \
       (~df["flag_not_cross_org"]) & (~df["flag_non_closed"])
post_counts = df.loc[keep, "L"].value_counts().sort_index()

Ls = sorted(pre_counts.index.tolist())
fig, ax = plt.subplots(figsize=(6.5, 3.0))
import numpy as np
xs = np.arange(len(Ls))
w = 0.42
ax.bar(xs - w/2, [pre_counts.get(L, 0) for L in Ls], width=w,
       label="pre-R6 (Phase-6 EW = 464,073)",
       color="#5A8FBC", edgecolor="black", linewidth=0.4)
ax.bar(xs + w/2, [post_counts.get(L, 0) for L in Ls], width=w,
       label="post-R1...R6 closed (84,564)",
       color="#E08030", edgecolor="black", linewidth=0.4)
ax.set_xticks(xs)
ax.set_xticklabels([str(L) for L in Ls])
ax.set_xlabel("slice length $L$")
ax.set_ylabel("patterns")
ax.set_yscale("log")
ax.set_title("Pattern count by slice length, before and after the R1–R6 filter chain")
ax.legend(frameon=False, loc="upper right")
ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_L_distribution.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

# ----------------------------------------------------------------------
# Figure 2: mechanism distribution (Phase 8 vs rule-based)
# ----------------------------------------------------------------------
with (ANA / "mechanism_classifier_metrics.json").open() as f:
    mech = json.load(f)
applied = mech["applied_to_ew_population"]
classes = ["background", "reusable_scenario", "shared_higher_level_step"]
xgb_vals = [applied["mechanism_xgb"][c] for c in classes]
rule_vals = [applied["mechanism_rule"][c] for c in classes]

fig, ax = plt.subplots(figsize=(6.5, 2.6))
xs = np.arange(len(classes))
w = 0.36
ax.bar(xs - w/2, xgb_vals, width=w, label="XGBoost (Phase 8)",
       color="#5A8FBC", edgecolor="black", linewidth=0.4)
ax.bar(xs + w/2, rule_vals, width=w, label="rule-based scope mapping",
       color="#9DBF94", edgecolor="black", linewidth=0.4)
for i, (xv, rv) in enumerate(zip(xgb_vals, rule_vals)):
    ax.text(i - w/2, xv + 5000, f"{xv:,}", ha="center", fontsize=7)
    ax.text(i + w/2, rv + 5000, f"{rv:,}", ha="center", fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(["background\n(RQ1)", "reusable\\_scenario\n(RQ2)",
                    "shared\\_higher\\_level\\_step\n(RQ3)"])
ax.set_ylabel("Phase-6 EW patterns assigned")
ax.set_title(f"Phase-8 mechanism distribution over the {applied['n_predicted_ew']:,}"
             f" predicted-EW patterns (XGBoost vs rule baseline; agreement = "
             f"{applied['rule_vs_xgb_agreement']*100:.1f}\\,\\%)")
ax.legend(frameon=False, loc="upper right")
ax.grid(True, axis="y", linestyle=":", linewidth=0.3, alpha=0.6)
ax.set_ylim(0, max(xgb_vals + rule_vals) * 1.15)
fig.tight_layout()
out = OUT / "fig_mech_distribution.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

# ----------------------------------------------------------------------
# Figure 3: classifier vs LLM judges
# ----------------------------------------------------------------------
with (ANA / "extraction_classifier_metrics.json").open() as f:
    cls = json.load(f)
with open(ROOT / "methodology" / "llm_judge_agreement.json") as f:
    llm = json.load(f)

raters = ["Phase-6 XGBoost", "gpt-oss-120b", "ling-2.6-1t"]
f1_yes = [
    cls["oof"]["f1"],
    llm["per_model"]["openai/gpt-oss-120b:free"]["vs_human"]["yes_class_f1"],
    llm["per_model"]["inclusionai/ling-2.6-1t:free"]["vs_human"]["yes_class_f1"],
]
acc = [
    (cls["oof"]["f1"]),  # placeholder; we'll show only F1 + kappa for the LLMs
    llm["per_model"]["openai/gpt-oss-120b:free"]["vs_human"]["accuracy_binary_yes_vs_rest"],
    llm["per_model"]["inclusionai/ling-2.6-1t:free"]["vs_human"]["accuracy_binary_yes_vs_rest"],
]
kappa_b = [
    None,  # not in artefact for classifier
    llm["per_model"]["openai/gpt-oss-120b:free"]["vs_human"]["cohen_kappa_binary"],
    llm["per_model"]["inclusionai/ling-2.6-1t:free"]["vs_human"]["cohen_kappa_binary"],
]
ci_lo = [cls["metrics_with_95ci"]["f1"]["ci_lo"], None, None]
ci_hi = [cls["metrics_with_95ci"]["f1"]["ci_hi"], None, None]

fig, ax = plt.subplots(figsize=(6.5, 2.8))
xs = np.arange(len(raters))
bars = ax.bar(xs, f1_yes,
              color=["#5A8FBC", "#B0B0B0", "#B0B0B0"],
              edgecolor="black", linewidth=0.4, width=0.55)
ax.errorbar(0, f1_yes[0],
            yerr=[[f1_yes[0] - ci_lo[0]], [ci_hi[0] - f1_yes[0]]],
            fmt="none", ecolor="black", capsize=3, elinewidth=0.8)
for i, v in enumerate(f1_yes):
    label = f"{v:.3f}"
    if i == 0:
        label += f"\n[{ci_lo[0]:.3f}, {ci_hi[0]:.3f}]"
    elif kappa_b[i] is not None:
        label += f"\n$\\kappa_b={kappa_b[i]:.3f}$"
    ax.text(i, v + 0.02, label, ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(raters)
ax.set_ylabel("$F_1$(yes) on the binary EW task")
ax.set_ylim(0, 1.05)
ax.set_title("Phase-6 classifier vs Phase-7 LLM judges on the same 200-slice pool"
             " (humans aggregated, 197 non-tie items)")
ax.grid(True, axis="y", linestyle=":", linewidth=0.3, alpha=0.6)
fig.tight_layout()
out = OUT / "fig_classifier_vs_llm.pdf"
fig.savefig(out)
plt.close(fig)
print(f"  wrote {out}")

print("done.")
