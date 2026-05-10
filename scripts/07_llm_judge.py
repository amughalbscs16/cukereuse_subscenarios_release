"""Phase 7 — LLM judge baseline.

Re-labels the 200-slice pool with two open-weights OpenRouter models
acting as independent judges under the same rubric the three authors
used. Reports per-model agreement with the human aggregated labels
(Cohen's kappa, accuracy, per-class precision/recall) — see
07b_llm_judge_agreement.py.

Models (all OpenRouter free tier):
  openai/gpt-oss-120b:free      (120B open-weights)
  inclusionai/ling-2.6-1t:free  (1T MoE)

Outputs:
  methodology/llm_judge/<model_slug>.jsonl
  methodology/llm_judge_summary.json
"""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "methodology" / "labels.jsonl"
OUT_DIR = ROOT / "methodology" / "llm_judge"
OUT_DIR.mkdir(exist_ok=True)
SUMMARY = ROOT / "methodology" / "llm_judge_summary.json"

KEY = open(ROOT / ".env", encoding="utf-8").read().strip().split("=", 1)[1]
SSL_CTX = ssl.create_default_context()

MODELS = [
    "openai/gpt-oss-120b:free",
    "inclusionai/ling-2.6-1t:free",
]

EXTRACT_CATS = ["yes", "no", "uncertain", "flagged-spec"]
MECH_CATS = ["background", "reusable_scenario",
             "shared_higher_level_step", "unsure", "n/a"]

SYSTEM_PROMPT = (
    "You are an experienced software-engineering researcher labelling "
    "candidate test-script subsequences (slices) for extraction-worthiness "
    "under a written rubric. Read the rubric carefully and apply it strictly. "
    "Output a single JSON object and nothing else. No markdown fences, no "
    "explanation outside the JSON."
)

RUBRIC_SUMMARY = """\
RUBRIC (paper 2 slice-level, condensed).

A SLICE is a contiguous L-step Gherkin sub-scenario (2 <= L <= 18) drawn
from a corpus of 1.1M BDD steps. Each slice carries scope signals:
  - max_within_file_recurrence: max # scenarios in any single .feature
    file containing this slice -> RQ1 / Background candidate signal
  - max_within_repo_files: max # files in any single repo containing the
    slice -> RQ2 / reusable_scenario signal
  - n_distinct_orgs: # distinct GitHub owners containing the slice
    -> RQ3 / shared_higher_level_step signal
  - outlier_fraction: fraction of occurrences on Phase-2c-flagged
    spec-suite files (high = generator artefact)

LABEL A: extraction_worthy in {yes, no, uncertain, flagged-spec}
  yes requires ALL FIVE:
    B-1 coherent: one logically grouped behaviour
    B-2 stable: no scenario-specific parameterisation that varies wildly
    B-3 non-trivial: meaningful behaviour, not a single Given promoted
        to L=2
    B-4 stand-alone: clear start/end; not cutting a When-Then binding
    B-5 genuine: not a generator artefact, outline expansion, or
        spec-suite duplication
  no fires on any of N-1..N-5:
    N-1 incoherent boundary
    N-2 domain-specific noise (recurring opaque value that won't
        generalise)
    N-3 already-extracted (already a Background or reusable elsewhere)
    N-4 too-short-for-value (L=2 with low semantic content)
    N-5 generated/outline duplication
  flagged-spec for generator artefacts: high outlier_fraction PLUS
    quoted-entity placeholders ("foo" "bar") or angle placeholders
    (<name>) suggesting templated generation.
  uncertain only when the rubric truly under-specifies; sparingly.

LABEL B: mechanism, conditional on extraction_worthy = yes:
  background (RQ1 only — within-file)
  reusable_scenario (RQ2 — within-repo cross-file)
  shared_higher_level_step (RQ3 — cross-organisational)
  unsure (yes but no clean mechanism fit)
  n/a (when extraction_worthy != yes)

CALIBRATION NOTES:
  - L=2 cross-org HTTP shorthand pairs (e.g. "method post" / "status 200")
    are borderline; default to no since extraction would replace 2 lines
    with 2 lines with no semantic gain.
  - Long L (>=10) slices that contain an obvious internal repetition
    of a short pattern: label yes/reusable_scenario but note the
    inner pattern would be a better extraction target.
  - Plain HTTP-header repetition WITHOUT placeholder structure is NOT
    flagged-spec; it can be a legitimate reusable_scenario candidate.
  - n_distinct_orgs is the corrected RQ3 metric (not n_distinct_repos),
    because one org publishing many language-specific SDK clones inflates
    the cross-repo count without representing genuine cross-org reuse.
"""


def call_openrouter(model: str, messages: list, max_tokens: int = 1024) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/amughalbscs16/cukereuse-subscenarios",
            "X-Title": "cukereuse-subscenarios paper-2 LLM judge",
        },
    )
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=90) as r:
        return json.load(r)


def call_with_retry(model: str, messages: list, max_tokens: int = 1024,
                    max_attempts: int = 6) -> dict | None:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_openrouter(model, messages, max_tokens)
        except urllib.error.HTTPError as e:
            last_err = (e.code, e.read().decode("utf-8", errors="replace")[:200])
            if e.code in (429, 502, 503, 504, 524):
                wait = min(2 ** attempt, 60)
                time.sleep(wait)
                continue
            return {"_error": last_err}
        except Exception as e:
            last_err = ("ERR", str(e)[:200])
            time.sleep(min(2 ** attempt, 60))
            continue
    return {"_error": last_err}


JSON_RE = re.compile(r"\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}", re.DOTALL)


def parse_verdict(content: str) -> dict:
    """Pull the JSON verdict out of the model's response. Tolerates
    markdown fences and accidental prose around the JSON."""
    if not content:
        return {"_parse_error": "empty content"}
    # Strip markdown fences
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Try direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Find first {...} substring
    for m in JSON_RE.finditer(s):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("extraction_worthy" in obj or "label_extraction_worthy" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return {"_parse_error": "no JSON found", "_raw": s[:300]}


def normalise_verdict(parsed: dict) -> tuple[str, str, str]:
    """Map a parsed verdict to (extract, mech, notes) using forgiving
    field names."""
    if "_parse_error" in parsed:
        return ("PARSE_FAIL", "PARSE_FAIL", parsed.get("_raw", "")[:200])
    ex = parsed.get("extraction_worthy") or parsed.get("label_extraction_worthy") or parsed.get("label_a") or ""
    me = parsed.get("mechanism") or parsed.get("label_mechanism") or parsed.get("label_b") or ""
    notes = parsed.get("notes") or parsed.get("rationale") or parsed.get("reason") or ""
    ex = str(ex).strip().lower().replace(" ", "-")
    me = str(me).strip().lower().replace(" ", "_")
    if ex not in EXTRACT_CATS:
        # tolerant matches
        if "flag" in ex and "spec" in ex:
            ex = "flagged-spec"
        elif ex.startswith("y"):
            ex = "yes"
        elif ex.startswith("n") and "n/a" not in ex:
            ex = "no"
        elif "uncert" in ex:
            ex = "uncertain"
        else:
            ex = "PARSE_FAIL"
    if me not in MECH_CATS:
        if "background" in me:
            me = "background"
        elif "reusable" in me:
            me = "reusable_scenario"
        elif "shared" in me or "higher" in me:
            me = "shared_higher_level_step"
        elif me in ("na", "none", ""):
            me = "n/a"
        elif "unsure" in me:
            me = "unsure"
        else:
            me = "PARSE_FAIL"
    return ex, me, str(notes)[:300]


def build_user_prompt(entry: dict) -> str:
    seq = entry["canonical_text_seq"]
    seq_lines = "\n".join(f"  [{i}] {str(t)[:160]}" for i, t in enumerate(seq))
    return (
        RUBRIC_SUMMARY
        + "\n\nSLICE TO LABEL:\n"
        + f"  L = {entry['L']}\n"
        + f"  scope = {entry['scope']} (most-specific RQ this slice qualifies for)\n"
        + f"  support_total = {entry['support_total']}\n"
        + f"  max_within_file_recurrence = {entry['max_within_file_recurrence']}\n"
        + f"  max_within_repo_files = {entry['max_within_repo_files']}\n"
        + f"  n_distinct_repos = {entry['n_distinct_repos']}\n"
        + f"  n_distinct_files = {entry['n_distinct_files']}\n"
        + f"  outlier_fraction = {entry.get('outlier_fraction', 0.0)}\n"
        + f"  example: {entry['example_repo']} / {entry['example_file']}\n"
        + f"  scenario: {entry['example_scenario']}\n"
        + f"  canonical_text_seq:\n{seq_lines}\n\n"
        + 'Output exactly this JSON shape, no extra text:\n'
        + '{"extraction_worthy": "<yes|no|uncertain|flagged-spec>", '
        + '"mechanism": "<background|reusable_scenario|shared_higher_level_step|unsure|n/a>", '
        + '"notes": "<one sentence, under 30 words, citing the deciding criterion (B-1..B-5 or N-1..N-5)>"}'
    )


def label_one_model(model: str, pool: list) -> dict:
    """Label all 200 entries with one model. Cache to disk per-entry
    so partial progress survives crashes."""
    slug = model.replace("/", "_").replace(":", "_")
    out = OUT_DIR / f"{slug}.jsonl"
    cache = {}
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["pattern"]] = rec
                except json.JSONDecodeError:
                    continue
    print(f"[{model}] cached: {len(cache)}/{len(pool)}")

    n_done = len(cache)
    n_fail = 0
    with out.open("a", encoding="utf-8") as fout:
        for i, entry in enumerate(pool):
            if entry["pattern"] in cache:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(entry)},
            ]
            t0 = time.time()
            res = call_with_retry(model, messages, max_tokens=1024)
            dt = time.time() - t0
            content = ""
            err = None
            if res is None or "_error" in res:
                err = res.get("_error") if isinstance(res, dict) else "no response"
                ex, me, notes = "API_FAIL", "API_FAIL", str(err)[:200]
            else:
                try:
                    msg = res["choices"][0]["message"]
                    content = msg.get("content", "") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    parsed = parse_verdict(content)
                    ex, me, notes = normalise_verdict(parsed)
                except Exception as e:
                    ex, me, notes = "PARSE_FAIL", "PARSE_FAIL", f"{type(e).__name__}: {str(e)[:200]}"
            rec = {
                "pattern": entry["pattern"],
                "L": entry["L"],
                "scope": entry["scope"],
                "stratum": entry["stratum"],
                "model": model,
                "extraction_worthy": ex,
                "mechanism": me,
                "notes": notes,
                "raw_content_excerpt": content[:300] if content else "",
                "wall_seconds": round(dt, 2),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if ex in ("API_FAIL", "PARSE_FAIL"):
                n_fail += 1
            if (i + 1) % 10 == 0 or i + 1 == len(pool):
                print(f"  [{model}] {i+1}/{len(pool)}  failures={n_fail}  last={ex}/{me}  ({dt:.1f}s)")
            # Spacing between calls within one model — even at temperature 0,
            # free tier dislikes >10 RPM
            time.sleep(2.5)
    return {"model": model, "n_done": n_done, "n_fail": n_fail, "out": str(out)}


def main() -> None:
    pool = [json.loads(l) for l in LABELS.open(encoding="utf-8")]
    # Drop the labels-internal fields the LLM doesn't need to see
    print(f"loaded {len(pool)} pool entries")

    # Run all 5 models concurrently — different providers, separate rate limits
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        futures = [ex.submit(label_one_model, m, pool) for m in MODELS]
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                r = {"model": "unknown", "_error": str(e)[:300]}
            results.append(r)
            print(f"FINISHED: {r}")

    # Aggregate per-model verdicts
    summary = {"per_model": {}}
    for m in MODELS:
        slug = m.replace("/", "_").replace(":", "_")
        f = OUT_DIR / f"{slug}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        ext = Counter(r["extraction_worthy"] for r in recs)
        mech = Counter(r["mechanism"] for r in recs)
        summary["per_model"][m] = {
            "n": len(recs),
            "extraction_worthy_distribution": dict(ext),
            "mechanism_distribution": dict(mech),
        }
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {SUMMARY}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
