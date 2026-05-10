"""Phase 7b — agreement analysis for the LLM-judge baseline.

For each of the five LLM judges, compute its agreement with the human
aggregated labels in methodology/labels.jsonl. Reports:
  - accuracy (binary: extraction_worthy == 'yes' vs not-yes)
  - 4-category accuracy (yes/no/uncertain/flagged-spec)
  - Cohen's kappa (binary and 4-cat)
  - Per-class precision/recall/F1 vs majority-human label
  - Inter-LLM Fleiss' kappa across the five judges

Also reports per-model failure rate (PARSE_FAIL + API_FAIL) so the
denominator for agreement is honestly reported.

Output: methodology/llm_judge_agreement.json
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

from sklearn.metrics import (cohen_kappa_score, classification_report,
                             precision_recall_fscore_support)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "methodology" / "labels.jsonl"
LLM_DIR = ROOT / "methodology" / "llm_judge"
OUT = ROOT / "methodology" / "llm_judge_agreement.json"

EXTRACT_CATS = ["yes", "no", "uncertain", "flagged-spec"]


def fleiss_kappa(items_labels: dict[str, list[str]], categories: list[str]) -> float:
    """Items dict: pattern -> list of labels (one per rater).
    All items must have the same number of raters.
    """
    if not items_labels:
        return float("nan")
    n_raters = len(next(iter(items_labels.values())))
    n_items = len(items_labels)
    cat_index = {c: i for i, c in enumerate(categories)}
    counts = [[0] * len(categories) for _ in range(n_items)]
    for i, (_, labs) in enumerate(items_labels.items()):
        for lab in labs:
            if lab in cat_index:
                counts[i][cat_index[lab]] += 1
    P_i = [
        (sum(c * c for c in row) - n_raters) / max(n_raters * (n_raters - 1), 1)
        for row in counts
    ]
    P_bar = sum(P_i) / n_items
    p_j = [sum(row[j] for row in counts) / max(n_items * n_raters, 1)
           for j in range(len(categories))]
    P_e = sum(p * p for p in p_j)
    if P_e >= 1.0 - 1e-12:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main() -> None:
    human = {r["pattern"]: r for r in (json.loads(l) for l in LABELS.open(encoding="utf-8"))}
    print(f"human labels: {len(human)}")

    summary = {"per_model": {}, "inter_llm_fleiss": {}}

    # Collect per-model verdicts
    all_models_labels = {}  # pattern -> {model -> label}
    for f in sorted(LLM_DIR.glob("*.jsonl")):
        if f.stat().st_size == 0:
            continue
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        if not recs:
            continue
        model = recs[0]["model"]
        n_pat = len(set(r["pattern"] for r in recs))
        # de-dup if any
        by_pat = {r["pattern"]: r for r in recs}

        # Failure rate
        fails = sum(1 for r in by_pat.values()
                    if r["extraction_worthy"] in ("PARSE_FAIL", "API_FAIL"))
        valid = [r for r in by_pat.values()
                 if r["extraction_worthy"] not in ("PARSE_FAIL", "API_FAIL")]
        print(f"\n{model}: n={len(by_pat)}, valid={len(valid)}, fails={fails}")

        # Distribution
        ext = Counter(r["extraction_worthy"] for r in valid)
        mech = Counter(r["mechanism"] for r in valid)
        print(f"  extraction_worthy: {dict(ext)}")
        print(f"  mechanism:         {dict(mech)}")

        # Agreement vs human (intersection of patterns where both have valid labels)
        common = []
        for r in valid:
            h = human.get(r["pattern"])
            if not h:
                continue
            if h["label_extraction_worthy"] in ("tie",):
                continue  # ties don't have a single human verdict
            common.append((h["label_extraction_worthy"], r["extraction_worthy"]))
        h_ext = [a for a, b in common]
        m_ext = [b for a, b in common]
        bin_h = [1 if x == "yes" else 0 for x in h_ext]
        bin_m = [1 if x == "yes" else 0 for x in m_ext]
        acc_4cat = sum(1 for a, b in zip(h_ext, m_ext) if a == b) / max(len(common), 1)
        acc_bin = sum(1 for a, b in zip(bin_h, bin_m) if a == b) / max(len(common), 1)
        kappa_4cat = cohen_kappa_score(h_ext, m_ext) if len(common) > 1 else float("nan")
        kappa_bin = cohen_kappa_score(bin_h, bin_m) if len(common) > 1 else float("nan")
        # Per-class precision/recall on binary task
        if sum(bin_h) > 0 and sum(bin_m) > 0:
            p, r_, f1, _ = precision_recall_fscore_support(
                bin_h, bin_m, labels=[1], zero_division=0)
            p_yes, r_yes, f1_yes = float(p[0]), float(r_[0]), float(f1[0])
        else:
            p_yes = r_yes = f1_yes = float("nan")
        # Mechanism agreement (only on yes-yes pairs where both said yes)
        common_yes = []
        for r in valid:
            h = human.get(r["pattern"])
            if not h: continue
            if h["label_extraction_worthy"] == "yes" and r["extraction_worthy"] == "yes":
                common_yes.append((h["label_mechanism"], r["mechanism"]))
        mech_acc = sum(1 for a, b in common_yes if a == b) / max(len(common_yes), 1)

        summary["per_model"][model] = {
            "n_total": len(by_pat),
            "n_valid": len(valid),
            "n_failed": fails,
            "extraction_worthy_distribution": dict(ext),
            "mechanism_distribution": dict(mech),
            "vs_human": {
                "n_compared": len(common),
                "accuracy_4cat": round(acc_4cat, 3),
                "accuracy_binary_yes_vs_rest": round(acc_bin, 3),
                "cohen_kappa_4cat": round(float(kappa_4cat), 3),
                "cohen_kappa_binary": round(float(kappa_bin), 3),
                "yes_class_precision": round(p_yes, 3),
                "yes_class_recall":    round(r_yes, 3),
                "yes_class_f1":        round(f1_yes, 3),
                "mechanism_accuracy_on_yes_yes": round(mech_acc, 3),
                "n_yes_yes": len(common_yes),
            },
        }

        # For inter-LLM Fleiss
        for r in valid:
            all_models_labels.setdefault(r["pattern"], {})[model] = r["extraction_worthy"]

    # Inter-LLM Fleiss kappa over patterns where all 5 valid
    n_models = sum(1 for f in LLM_DIR.glob("*.jsonl") if f.stat().st_size > 0)
    full_set = {p: list(d.values()) for p, d in all_models_labels.items()
                if len(d) == n_models}
    if full_set and n_models > 1:
        kappa_4cat = fleiss_kappa(full_set, EXTRACT_CATS)
        # Binary collapse
        binary_set = {p: ["yes" if v == "yes" else "not-yes" for v in vs]
                      for p, vs in full_set.items()}
        kappa_bin = fleiss_kappa(binary_set, ["yes", "not-yes"])
        summary["inter_llm_fleiss"] = {
            "n_models": n_models,
            "n_items_all_models_valid": len(full_set),
            "fleiss_kappa_4cat": round(float(kappa_4cat), 3),
            "fleiss_kappa_binary": round(float(kappa_bin), 3),
        }
        print(f"\ninter-LLM Fleiss kappa (4-cat):  {kappa_4cat:.3f}  on {len(full_set)} items")
        print(f"inter-LLM Fleiss kappa (binary): {kappa_bin:.3f}")

    # Best model = highest Cohen's kappa vs human (binary)
    if summary["per_model"]:
        ranked = sorted(
            summary["per_model"].items(),
            key=lambda kv: kv[1]["vs_human"]["cohen_kappa_binary"],
            reverse=True,
        )
        summary["ranking_by_cohen_kappa_binary"] = [
            {"model": m, "kappa": d["vs_human"]["cohen_kappa_binary"],
             "f1_yes": d["vs_human"]["yes_class_f1"]}
            for m, d in ranked
        ]
        print("\n=== ranking by Cohen kappa vs human (binary yes/not-yes) ===")
        for r in summary["ranking_by_cohen_kappa_binary"]:
            print(f"  {r['model']:55s}  kappa={r['kappa']:+.3f}  F1(yes)={r['f1_yes']:.3f}")

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
