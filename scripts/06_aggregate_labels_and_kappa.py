"""Phase 6 prep — aggregate three-author labels and compute Fleiss' kappa.

Inputs:
  methodology/labels_in_progress/labels_author_A.jsonl  (106 records)
  methodology/labels_in_progress/labels_author_B.jsonl  (106 records)
  methodology/labels_in_progress/labels_author_C.jsonl  (108 records)

Outputs:
  methodology/labels.jsonl
      200 records, one per pool entry. Each record carries:
        - all pool metadata
        - labels_per_author: dict keyed by author_id -> {extraction_worthy, mechanism, notes}
        - label_extraction_worthy: majority verdict (or 'tie' if all 3 differ)
        - label_mechanism: majority verdict (n/a where extraction_worthy != yes)
        - n_authors: 3 for overlap, 1 for own entries
  methodology/labels_summary.json
      Aggregate stats: distribution by author, agreement on overlap,
      Fleiss' kappa on the 4-category extraction-worthy label, Cohen's
      pairwise agreement, joint distributions.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "methodology" / "labels_in_progress"
OUT_LABELS = ROOT / "methodology" / "labels.jsonl"
OUT_SUMMARY = ROOT / "methodology" / "labels_summary.json"

EXTRACT_CATS = ["yes", "no", "uncertain", "flagged-spec"]


def load(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def fleiss_kappa(pattern_labels: dict[str, list[str]], categories: list[str]) -> float:
    """Fleiss' kappa across N items rated by a fixed number of raters.

    pattern_labels: pattern_id -> [labels from N raters]; expects same N
    for every item. Returns kappa in [-1, 1].
    """
    if not pattern_labels:
        return float("nan")
    n_raters = len(next(iter(pattern_labels.values())))
    n_items = len(pattern_labels)
    cat_index = {c: i for i, c in enumerate(categories)}
    counts = [[0] * len(categories) for _ in range(n_items)]
    for i, (_, labs) in enumerate(pattern_labels.items()):
        for lab in labs:
            counts[i][cat_index[lab]] += 1
    P_i = [
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in counts
    ]
    P_bar = sum(P_i) / n_items
    p_j = [sum(row[j] for row in counts) / (n_items * n_raters) for j in range(len(categories))]
    P_e = sum(p * p for p in p_j)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohen_pairwise(a: list[str], b: list[str]) -> float:
    """Simple agreement % (not chi-squared kappa) for paper sanity."""
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def majority(labels: list[str]) -> str:
    c = Counter(labels)
    top, top_n = c.most_common(1)[0]
    if list(c.values()).count(top_n) > 1:
        return "tie"
    return top


def main() -> None:
    A = load(LABELS_DIR / "labels_author_A.jsonl")
    B = load(LABELS_DIR / "labels_author_B.jsonl")
    C = load(LABELS_DIR / "labels_author_C.jsonl")

    by_pattern = defaultdict(dict)
    for au, recs in [("A", A), ("B", B), ("C", C)]:
        for r in recs:
            by_pattern[r["pattern"]][au] = r
    print(f"merged: {len(by_pattern)} unique patterns")

    # -------- per-author distribution --------
    dist_per_author = {}
    for au, recs in [("A", A), ("B", B), ("C", C)]:
        dist_per_author[au] = {
            "extraction_worthy": dict(Counter(r["label_extraction_worthy"] for r in recs)),
            "mechanism": dict(Counter(r["label_mechanism"] for r in recs)),
            "n": len(recs),
        }

    # -------- overlap-only Fleiss & pairwise --------
    overlap_patterns = {p for p, m in by_pattern.items() if len(m) == 3}
    print(f"overlap (3-way) patterns: {len(overlap_patterns)}")

    overlap_labels_extract = {
        p: [by_pattern[p][au]["label_extraction_worthy"] for au in ("A", "B", "C")]
        for p in overlap_patterns
    }
    kappa_extract = fleiss_kappa(overlap_labels_extract, EXTRACT_CATS)

    # mechanism: only well-defined when all three said yes; otherwise treat n/a as a category
    MECH_CATS = ["background", "reusable_scenario", "shared_higher_level_step",
                 "unsure", "n/a"]
    overlap_labels_mech = {
        p: [by_pattern[p][au]["label_mechanism"] for au in ("A", "B", "C")]
        for p in overlap_patterns
    }
    kappa_mech = fleiss_kappa(overlap_labels_mech, MECH_CATS)

    # pairwise agreement
    pair_agree = {}
    for x, y in [("A", "B"), ("A", "C"), ("B", "C")]:
        a = [by_pattern[p][x]["label_extraction_worthy"] for p in overlap_patterns]
        b = [by_pattern[p][y]["label_extraction_worthy"] for p in overlap_patterns]
        pair_agree[f"{x}-{y}_extract"] = round(cohen_pairwise(a, b), 3)
        a = [by_pattern[p][x]["label_mechanism"] for p in overlap_patterns]
        b = [by_pattern[p][y]["label_mechanism"] for p in overlap_patterns]
        pair_agree[f"{x}-{y}_mech"] = round(cohen_pairwise(a, b), 3)

    # majority verdicts on overlap
    majority_extract = Counter()
    tie_examples = []
    for p in overlap_patterns:
        m = majority(overlap_labels_extract[p])
        majority_extract[m] += 1
        if m == "tie":
            tie_examples.append({"pattern": p, "labels": overlap_labels_extract[p]})

    # -------- write final labels.jsonl --------
    written = 0
    with OUT_LABELS.open("w", encoding="utf-8") as fout:
        for p, m in by_pattern.items():
            base = next(iter(m.values())).copy()
            for k in ("label_extraction_worthy", "label_mechanism",
                      "labeller_notes", "labeller_id", "_subset"):
                base.pop(k, None)
            base["labels_per_author"] = {
                au: {
                    "extraction_worthy": m[au]["label_extraction_worthy"],
                    "mechanism": m[au]["label_mechanism"],
                    "notes": m[au].get("labeller_notes", ""),
                }
                for au in m
            }
            base["n_authors"] = len(m)
            if len(m) == 3:
                ex = [m[au]["label_extraction_worthy"] for au in ("A", "B", "C")]
                me = [m[au]["label_mechanism"] for au in ("A", "B", "C")]
                base["label_extraction_worthy"] = majority(ex)
                base["label_mechanism"] = majority(me)
            else:
                # single author for non-overlap
                au = next(iter(m))
                base["label_extraction_worthy"] = m[au]["label_extraction_worthy"]
                base["label_mechanism"] = m[au]["label_mechanism"]
            fout.write(json.dumps(base, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {OUT_LABELS} : {written} records")

    # -------- summary --------
    summary = {
        "n_total": written,
        "n_overlap": len(overlap_patterns),
        "per_author_distribution": dist_per_author,
        "fleiss_kappa_extraction_worthy_4cat": round(kappa_extract, 3),
        "fleiss_kappa_mechanism_5cat": round(kappa_mech, 3),
        "pairwise_agreement_overlap": pair_agree,
        "majority_distribution_overlap_extract": dict(majority_extract),
        "tie_examples": tie_examples[:5],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
