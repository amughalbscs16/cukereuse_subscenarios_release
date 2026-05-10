"""Pull a diverse 10-entry sample from the labelling pool for pilot labelling.

Goal: surface entries that exercise different rubric cells so the user
can spot-check my draft labels before authors invest time.

Picks:
  - 2 high-confidence RQ3 cross-repo yes candidates (L>=4, multi-repo)
  - 2 RQ2 within-repo yes candidates
  - 2 RQ1 within-file Background candidates (different L)
  - 1 short L=2 weak candidate (likely no)
  - 1 long L>=10 candidate (rare; check if coherent)
  - 1 spec_dominated entry (flagged-spec test)
  - 1 borderline / uncertain (low support, mid L)
"""

from __future__ import annotations

import json
import pathlib

POOL = pathlib.Path(__file__).resolve().parent.parent / "methodology" / "labeling_pool.jsonl"


def load() -> list[dict]:
    with POOL.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick(pool: list[dict]) -> list[tuple[str, dict]]:
    picks: list[tuple[str, dict]] = []

    # 2x RQ3 cross-repo, L >= 4, real_signal, support >= 6
    rq3 = [r for r in pool if r["stratum"] == "real_signal"
           and r["scope"] == "RQ3" and r["L"] >= 4
           and r["support_total"] >= 6]
    rq3.sort(key=lambda r: (r["n_distinct_repos"], r["L"]), reverse=True)
    for r in rq3[:2]:
        picks.append(("RQ3-cross-repo", r))

    # 2x RQ2 within-repo, real_signal, max_within_repo_files >= 3
    rq2 = [r for r in pool if r["stratum"] == "real_signal"
           and r["scope"] == "RQ2" and r["max_within_repo_files"] >= 3
           and r["L"] >= 3]
    rq2.sort(key=lambda r: r["max_within_repo_files"], reverse=True)
    for r in rq2[:2]:
        picks.append(("RQ2-within-repo", r))

    # 2x RQ1 within-file, real_signal, different L
    rq1 = [r for r in pool if r["stratum"] == "real_signal"
           and r["scope"] == "RQ1" and r["max_within_file_recurrence"] >= 2]
    rq1.sort(key=lambda r: r["max_within_file_recurrence"], reverse=True)
    chosen_L: set[int] = set()
    for r in rq1:
        if r["L"] not in chosen_L:
            picks.append(("RQ1-within-file", r))
            chosen_L.add(r["L"])
        if len(chosen_L) == 2:
            break

    # 1x short L=2, low support — likely weak
    weak = [r for r in pool if r["L"] == 2 and r["support_total"] <= 5
            and r["stratum"] == "real_signal"]
    if weak:
        picks.append(("L=2-weak-candidate", weak[0]))

    # 1x long L >= 10
    longish = [r for r in pool if r["L"] >= 10
               and r["stratum"] == "real_signal"]
    if longish:
        longish.sort(key=lambda r: r["L"], reverse=True)
        picks.append(("long-L", longish[0]))

    # 1x spec_dominated
    spec = [r for r in pool if r["stratum"] == "spec_dominated"]
    if spec:
        spec.sort(key=lambda r: r["support_total"], reverse=True)
        picks.append(("spec-dominated", spec[0]))

    # 1x borderline: mid-L, mid-support, real_signal
    border = [r for r in pool if r["stratum"] == "real_signal"
              and 4 <= r["L"] <= 7 and 5 <= r["support_total"] <= 20]
    if border:
        picks.append(("borderline", border[0]))

    return picks


def fmt_entry(tag: str, e: dict, idx: int) -> str:
    lines = [f"=== Entry #{idx} ({tag}) ==="]
    lines.append(f"  L = {e['L']}, scope = {e['scope']}, stratum = {e['stratum']}, "
                 f"outlier_fraction = {e['outlier_fraction']}")
    lines.append(f"  support_total = {e['support_total']:,}, "
                 f"n_distinct_repos = {e['n_distinct_repos']}, "
                 f"n_distinct_files = {e['n_distinct_files']}")
    lines.append(f"  max_within_file_recurrence = {e['max_within_file_recurrence']:,}, "
                 f"max_within_repo_files = {e['max_within_repo_files']:,}")
    lines.append(f"  example: {e['example_repo']}  /  {e['example_file']}")
    lines.append(f"           scenario: {e['example_scenario']}")
    lines.append("  canonical_text_seq:")
    for i, t in enumerate(e["canonical_text_seq"]):
        s = str(t)
        if len(s) > 110:
            s = s[:110] + "..."
        lines.append(f"     [{i}] {s}")
    return "\n".join(lines)


if __name__ == "__main__":
    pool = load()
    picks = pick(pool)
    for i, (tag, e) in enumerate(picks, 1):
        print(fmt_entry(tag, e, i))
        print()
