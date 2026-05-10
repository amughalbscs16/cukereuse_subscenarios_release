"""Phase 11 — produce a comprehensive, *quality-filtered* HTML report on
the extraction-worthy candidates so the result set can be inspected
end-to-end and any single pattern's full call-site list verified.

This version applies a transparent QA filter chain to suppress
known-degenerate candidate classes that survive the Phase-6 classifier
but are not genuinely valuable as reuse targets:

  R1 templated-outline    : any canonical step contains an
                             angle-bracket placeholder <\\w+>
                             (Scenario-Outline expansion residue;
                             v3 spec-suite detector misses these
                             because it only flags quote-pair
                             placeholders).
  R2 repetition           : cluster_id_seq is a single cluster
                             repeated (a,a,a,...,a) — assertion
                             list rather than reusable run.
  R3 single-scenario      : n_distinct_scenarios = 1 — the entire
                             support_total comes from overlapping
                             windows in one scenario, not reuse.
  R4 overlap-dominated    : n_distinct_scenarios / support_total
                             < 0.20 — most of the support is
                             overlapping windows within a small
                             number of scenarios.
  R5 mechanism-mismatch   : shared_higher_level_step requires
                             n_distinct_orgs >= 2 (rubric RQ3
                             definition).
  R6 non-closed           : pattern P is dropped if there exists a
                             super-pattern Q with cluster_id_seq
                             prefix-extension or suffix-extension of
                             P and support(Q) == support(P) AND
                             same scope. This is the standard closed
                             sequential pattern criterion: P is
                             redundant because every occurrence of P
                             is also an occurrence of Q.

Surviving candidates are ranked by a spread-aware quality score:
   q = n_distinct_scenarios * sqrt(support_total)
for background and reusable_scenario; for shared_higher_level_step
   q = n_distinct_orgs * n_distinct_scenarios * sqrt(support_total).
This rewards patterns that recur across diverse scenarios over
patterns that recur via overlapping windows in a few scenarios.

Outputs:
  analysis/extraction_candidates_report.html  (main verification report)
  analysis/extraction_candidates_index.csv    (full 464k post-classifier index, unfiltered)
  analysis/extraction_candidates_filtered.csv (post-QA-filter index with quality score)
"""

from __future__ import annotations

import csv
import html
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANA = ROOT / "analysis"
OUT_HTML = ANA / "extraction_candidates_report.html"
OUT_CSV_FULL = ANA / "extraction_candidates_index.csv"
OUT_CSV_FILT = ANA / "extraction_candidates_filtered.csv"

TOP_N_PER_MECH = 500     # patterns rendered with full call-site lists
                          # (per mechanism, post-filter, top of quality ranking)
MAX_SITES_PER_PATTERN = 50    # call sites inlined per pattern

ANGLE_PLACEHOLDER = re.compile(r"<\w+(-\w+)*>")

# ----------------------------------------------------------------------
# Load classifier output + canonical text
# ----------------------------------------------------------------------
print("loading mechanism_predictions...", flush=True)
mech = pd.read_parquet(ANA / "mechanism_predictions.parquet")
print(f"  {len(mech):,} EW candidates", flush=True)

print("loading exact_subsequence_ranking...", flush=True)
rank = pd.read_parquet(ANA / "exact_subsequence_ranking.parquet")
rank_min = rank[["pattern", "canonical_text_seq", "n_distinct_files",
                 "n_distinct_scenarios", "has_template_structure"]]
mech = mech.merge(rank_min, on="pattern", how="left")
print(f"  joined canonical text + scenario count", flush=True)

# ----------------------------------------------------------------------
# Compute QA flags
# ----------------------------------------------------------------------
print("computing QA flags ...", flush=True)

def has_angle_placeholder(seq) -> bool:
    if seq is None:
        return False
    for s in seq:
        if s and ANGLE_PLACEHOLDER.search(str(s)):
            return True
    return False

def is_repetition(pat: str) -> bool:
    parts = pat.split(",")
    return len(set(parts)) == 1

mech["flag_templated"]      = mech["canonical_text_seq"].apply(has_angle_placeholder)
mech["flag_repetition"]     = mech["pattern"].apply(is_repetition)
mech["flag_single_scen"]    = mech["n_distinct_scenarios"] <= 1
mech["scen_ratio"]          = (mech["n_distinct_scenarios"]
                                / mech["support_total"].clip(lower=1))
mech["flag_overlap_dom"]    = mech["scen_ratio"] < 0.20

# Mechanism-aware mismatch: shared_higher_level_step requires >=2 orgs
mech["flag_not_cross_org"]  = (
    (mech["mechanism"] == "shared_higher_level_step")
    & (mech["n_distinct_orgs"] < 2)
)

# Closed-pattern check (R6).
# A pattern P is non-closed if a super-pattern Q exists with the same
# scope and same support_total, where Q's cluster_id_seq is either
# prefix(P) extended on the right or suffix(P) extended on the left.
# This is a one-step closure check (it catches near-redundant
# length+1 super-patterns) which is sufficient because longer chains
# of slice-redundancy are then collapsed transitively as we work
# upward in L.
print("computing closure (R6) ...", flush=True)
# Build index: (scope, support, cluster_id_seq_str) -> True for every pattern
key_set = set()
for r in mech.itertuples(index=False):
    key_set.add((r.scope, r.support_total, r.pattern))
flag_non_closed = []
for r in mech.itertuples(index=False):
    pat = r.pattern
    parts = pat.split(",")
    L = len(parts)
    if L >= 18:
        flag_non_closed.append(False)
        continue
    s, sup = r.scope, r.support_total
    is_redundant = False
    # right-extension: prepend any clusterless wildcard. We check
    # against every distinct cluster id seen as the "next" cluster
    # in any L+1 pattern starting with `pat` -- but iterating clusters
    # is O(V). Use a presence index built from longer patterns instead.
    flag_non_closed.append(False)  # placeholder; filled below
mech["flag_non_closed"] = flag_non_closed

# Build a more efficient closure index: group L+1 patterns by their
# (scope, support_total). For each L pattern, look up whether any L+1
# pattern at the same (scope, support_total) extends it.
print("  building length-bucketed index ...", flush=True)
by_L = {L: mech[mech["L"] == L][["pattern", "scope", "support_total"]] for L in mech["L"].unique()}

# For each (scope, support) bucket at length L+1, prepare a set of
# the substrings (pattern[:-1] cluster_id_seq) and (pattern[1:]
# cluster_id_seq) so a length-L pattern can ask "do you contain me
# as a prefix or suffix at the same support?"
print("  computing prefix/suffix sets per (scope, support, L+1) ...", flush=True)
prefix_index: dict = {}     # (scope, support, L+1) -> set of pattern[:-1] strings
suffix_index: dict = {}     # (scope, support, L+1) -> set of pattern[1:] strings
for L_plus_1, df_l1 in by_L.items():
    for r in df_l1.itertuples(index=False):
        parts = r.pattern.split(",")
        if len(parts) < 2:
            continue
        prefix = ",".join(parts[:-1])
        suffix = ",".join(parts[1:])
        key = (r.scope, r.support_total, L_plus_1)
        prefix_index.setdefault(key, set()).add(prefix)
        suffix_index.setdefault(key, set()).add(suffix)

print("  marking non-closed patterns ...", flush=True)
non_closed_flags = []
for r in mech.itertuples(index=False):
    L = r.L
    key_l1 = (r.scope, r.support_total, L + 1)
    nc = (
        r.pattern in prefix_index.get(key_l1, ())
        or r.pattern in suffix_index.get(key_l1, ())
    )
    non_closed_flags.append(nc)
mech["flag_non_closed"] = non_closed_flags

# Aggregate reject flag
mech["any_flag"] = (
    mech["flag_templated"]
    | mech["flag_repetition"]
    | mech["flag_single_scen"]
    | mech["flag_overlap_dom"]
    | mech["flag_not_cross_org"]
    | mech["flag_non_closed"]
)

# Quality score (spread-aware)
def quality_score(row) -> float:
    base = row["n_distinct_scenarios"] * math.sqrt(max(row["support_total"], 1))
    if row["mechanism"] == "shared_higher_level_step":
        base *= max(row["n_distinct_orgs"], 1)
    return float(base)

mech["quality_score"] = mech.apply(quality_score, axis=1)

# Rejection reason (human-readable, first reason only for at-a-glance)
def rejection_reason(row) -> str:
    if row["flag_templated"]:    return "R1 templated-outline"
    if row["flag_repetition"]:   return "R2 repetition"
    if row["flag_single_scen"]:  return "R3 single-scenario"
    if row["flag_overlap_dom"]:  return "R4 overlap-dominated"
    if row["flag_not_cross_org"]:return "R5 mechanism-mismatch"
    if row["flag_non_closed"]:   return "R6 non-closed"
    return ""

mech["rejection_reason"] = mech.apply(rejection_reason, axis=1)

# ----------------------------------------------------------------------
# Filter funnel stats
# ----------------------------------------------------------------------
n0 = len(mech)
funnel = []
def step(name, mask):
    nbef = funnel[-1][1] if funnel else n0
    naft = (~mask).sum()
    funnel.append((name, naft, int(mask.sum()), nbef - naft))

# Apply each filter cumulatively, in order
working = mech.copy()
m_t  = working["flag_templated"]
working = working[~m_t]
funnel.append(("After R1: drop angle-bracket templated outlines",
               len(working), int(m_t.sum())))

m_r  = working["flag_repetition"]
working = working[~m_r]
funnel.append(("After R2: drop single-cluster repetition runs",
               len(working), int(m_r.sum())))

m_s  = working["flag_single_scen"]
working = working[~m_s]
funnel.append(("After R3: drop single-scenario patterns",
               len(working), int(m_s.sum())))

m_o  = working["flag_overlap_dom"]
working = working[~m_o]
funnel.append(("After R4: drop overlap-dominated (scen/support < 0.20)",
               len(working), int(m_o.sum())))

m_x  = working["flag_not_cross_org"]
working = working[~m_x]
funnel.append(("After R5: drop SHL with <2 distinct orgs",
               len(working), int(m_x.sum())))

m_c  = working["flag_non_closed"]
working = working[~m_c]
funnel.append(("After R6: drop non-closed (length+1 super-pattern with same support)",
               len(working), int(m_c.sum())))

filtered = working
print(f"funnel: {n0:,} -> {len(filtered):,}", flush=True)
for name, n, dropped in funnel:
    print(f"  {name}: {n:,} (dropped {dropped:,})")

# ----------------------------------------------------------------------
# Pick the top-N per mechanism by quality_score (post-filter)
# ----------------------------------------------------------------------
def topN(df, n):
    return df.sort_values(
        ["quality_score", "support_total"], ascending=[False, False]
    ).head(n)

top_per_mech = {
    "background":               topN(filtered[filtered["mechanism"] == "background"], TOP_N_PER_MECH),
    "reusable_scenario":        topN(filtered[filtered["mechanism"] == "reusable_scenario"], TOP_N_PER_MECH),
    "shared_higher_level_step": topN(filtered[filtered["mechanism"] == "shared_higher_level_step"], TOP_N_PER_MECH),
}

target_patterns = set()
for df in top_per_mech.values():
    target_patterns.update(df["pattern"].tolist())
print(f"target patterns (full call-site detail): {len(target_patterns):,}", flush=True)

# ----------------------------------------------------------------------
# Single-pass pyarrow scan over slices.parquet
# ----------------------------------------------------------------------
print("scanning slices.parquet ...", flush=True)
all_pats_set = set(filtered["pattern"].tolist())
target_set = target_patterns

repo_set_per_pattern: dict[str, set[str]] = defaultdict(set)
sites_per_pattern: dict[str, list[dict]] = defaultdict(list)

pf = pq.ParquetFile(ANA / "slices.parquet")
scanned = 0
for batch in pf.iter_batches(
    batch_size=200_000,
    columns=["repo_slug", "file_path", "scenario",
             "position_start", "cluster_id_seq", "text_seq"],
):
    cseq = batch.column("cluster_id_seq").to_pylist()
    repo = batch.column("repo_slug").to_pylist()
    fp   = batch.column("file_path").to_pylist()
    scen = batch.column("scenario").to_pylist()
    pos  = batch.column("position_start").to_pylist()
    txt  = batch.column("text_seq").to_pylist()
    for i in range(len(cseq)):
        pat = ",".join(map(str, cseq[i]))
        if pat in all_pats_set:
            repo_set_per_pattern[pat].add(repo[i])
            if pat in target_set:
                lst = sites_per_pattern[pat]
                if len(lst) < MAX_SITES_PER_PATTERN:
                    lst.append({
                        "repo": repo[i],
                        "file": fp[i],
                        "scen": scen[i],
                        "pos":  int(pos[i]),
                        "txt":  list(txt[i]),
                    })
    scanned += len(cseq)
    if scanned % 1_000_000 < 200_000:
        print(f"  scanned {scanned:,} rows", flush=True)

repo_candidate_counts = Counter()
for pat, repos in repo_set_per_pattern.items():
    for r in repos:
        repo_candidate_counts[r] += 1
top10_repos = repo_candidate_counts.most_common(10)
n_repos = len(repo_candidate_counts)
print(f"distinct repos hosting a *kept* candidate: {n_repos:,}", flush=True)

# ----------------------------------------------------------------------
# Write CSV indices
# ----------------------------------------------------------------------
print("writing CSV indices ...", flush=True)
csv_cols_full = ["pattern", "L", "scope", "mechanism",
                 "support_total", "n_distinct_scenarios",
                 "n_distinct_files", "n_distinct_repos", "n_distinct_orgs",
                 "max_within_file_recurrence", "max_within_repo_files",
                 "outlier_fraction", "p_extraction_worthy",
                 "scen_ratio", "quality_score",
                 "flag_templated", "flag_repetition", "flag_single_scen",
                 "flag_overlap_dom", "flag_not_cross_org", "flag_non_closed",
                 "rejection_reason", "has_template_structure"]
mech.sort_values(
    ["mechanism", "quality_score"], ascending=[True, False],
)[csv_cols_full].to_csv(OUT_CSV_FULL, index=False)

csv_cols_filt = [c for c in csv_cols_full
                 if c not in ("flag_templated", "flag_repetition",
                              "flag_single_scen", "flag_overlap_dom",
                              "flag_not_cross_org", "flag_non_closed",
                              "rejection_reason")]
filtered.sort_values(
    ["mechanism", "quality_score"], ascending=[True, False],
)[csv_cols_filt].to_csv(OUT_CSV_FILT, index=False)

# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""

def render_pattern_detail(row, idx: int) -> str:
    pat = row["pattern"]
    canon = row["canonical_text_seq"]
    canon_html = (
        "<ol class='steps'>" + "".join(
            f"<li>{esc(s)}</li>" for s in canon
        ) + "</ol>"
        if isinstance(canon, list) or hasattr(canon, "__iter__") and not isinstance(canon, str)
        else f"<pre>{esc(canon)}</pre>"
    )

    sites = sites_per_pattern.get(pat, [])
    n_repos_for_pat = len(repo_set_per_pattern.get(pat, set()))
    sites_html = []
    for s in sites:
        site_steps = "".join(
            f"<li>{esc(t)}</li>" for t in s["txt"]
        )
        sites_html.append(
            f"<div class='site'>"
            f"<div class='site-head'>"
            f"<span class='repo'>{esc(s['repo'])}</span>"
            f"<span class='sep'>/</span>"
            f"<span class='file mono'>{esc(s['file'])}</span>"
            f"<span class='sep'>::</span>"
            f"<span class='scenario'>{esc(s['scen'])}</span>"
            f"<span class='pos'>(starts at scenario step #{s['pos']+1})</span>"
            f"</div>"
            f"<ol start='{s['pos']+1}' class='site-steps'>{site_steps}</ol>"
            f"</div>"
        )
    if len(sites) == MAX_SITES_PER_PATTERN and row["support_total"] > MAX_SITES_PER_PATTERN:
        sites_html.append(
            f"<div class='cap-note'>(showing first {MAX_SITES_PER_PATTERN} "
            f"call sites; pattern has {row['support_total']} total slice "
            f"instances across {row['n_distinct_scenarios']} distinct scenarios)</div>"
        )
    sites_block = "<div class='sites'>" + "".join(sites_html) + "</div>"
    if not sites:
        sites_block = "<div class='no-sites'>(no example sites indexed)</div>"

    return (
        f"<tr class='row' data-idx='{idx}' data-pattern='{esc(pat)}'>"
        f"<td class='c-num'>{idx + 1}</td>"
        f"<td class='c-num'><b>{row['quality_score']:.1f}</b></td>"
        f"<td class='c-pat mono'>{esc(pat[:36])}{'…' if len(pat) > 36 else ''}</td>"
        f"<td class='c-num'>{row['L']}</td>"
        f"<td>{esc(row['scope'])}</td>"
        f"<td class='c-num'>{row['support_total']}</td>"
        f"<td class='c-num'><b>{row['n_distinct_scenarios']}</b></td>"
        f"<td class='c-num'>{row['scen_ratio']:.2f}</td>"
        f"<td class='c-num'>{n_repos_for_pat}</td>"
        f"<td class='c-num'>{row.get('n_distinct_orgs', 0)}</td>"
        f"<td class='c-num'>{row['n_distinct_files']}</td>"
        f"<td class='c-num'>{row['p_extraction_worthy']:.3f}</td>"
        f"</tr>"
        f"<tr class='detail' data-for='{idx}' style='display:none'>"
        f"<td colspan='12'>"
        f"<div class='detail-body'>"
        f"<h4>Representative slice text "
        f"<span class='hint'>(literal text of one observed slice; the cluster identity, "
        f"not the literal text, is what defines the pattern)</span></h4>"
        f"{canon_html}"
        f"<h4>Recorded call sites &mdash; {len(sites)} of "
        f"<b>{row['support_total']}</b> total slice instances "
        f"across <b>{row['n_distinct_scenarios']}</b> distinct scenarios "
        f"in {n_repos_for_pat} repo{'s' if n_repos_for_pat != 1 else ''}</h4>"
        f"{sites_block}"
        f"</div></td></tr>"
    )

def render_table(df, table_id: str) -> str:
    rows = "".join(
        render_pattern_detail(r, i) for i, (_, r) in enumerate(df.iterrows())
    )
    return (
        f"<table id='{table_id}' class='ranked'>"
        f"<thead><tr>"
        f"<th>#</th><th>q-score</th><th>pattern</th><th>L</th>"
        f"<th>scope</th><th>support</th>"
        f"<th>scenarios</th><th>scen/sup</th>"
        f"<th>repos</th><th>orgs</th>"
        f"<th>files</th><th>p(EW)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )

# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
sections = []

sections.append(f"""
<section id='primer'>
  <h2>1. What this report shows</h2>
  <p>The <b>cukereuse-subscenarios</b> discovery pipeline mines the
  <b>1,113,616-step</b> Gherkin corpus (paper~1) into <b>5,382,249</b>
  contiguous L-step <i>slices</i> (every window
  <code>(j, j+1, &hellip;, j+L-1)</code> for L in [2,18] and every
  start position j). After paraphrase clustering (Phase~4) those
  slices map onto <b>692,020</b> distinct <i>patterns</i> (a pattern
  is a cluster-id sequence). The Phase-6 binary classifier predicts
  <b>464,073</b> of them as <i>extraction-worthy</i> and the Phase-8
  head assigns each to one of three Mughal-2024 reuse mechanisms.</p>
  <p><b>This report applies a transparent QA filter to remove known
  false-positive and redundant patterns</b> &mdash; templated
  Scenario-Outline expansions (angle-bracket placeholders), single-
  cluster repetition runs, single-scenario patterns, overlap-dominated
  patterns, mechanism-mismatched cross-org claims, and
  <i>non-closed</i> patterns (a length-L pattern dropped if a
  length-(L+1) super-pattern with the same support exists, collapsing
  slice variants of the same underlying reuse opportunity). The funnel
  is shown below; the surviving patterns are ranked by a spread-aware
  quality score
  <code>q = n_distinct_scenarios &times; sqrt(support_total)</code>
  (multiplied by <code>n_distinct_orgs</code> for cross-org candidates)
  so that patterns recurring across diverse scenarios outrank patterns
  that recur via overlapping windows in a few long scenarios.</p>
  <p>Click any row to expand a pattern's representative step text and
  the recorded call sites. Each call site shows
  <code>repo / file :: scenario (step #X)</code>: that 4-tuple plus
  the L step texts is exactly enough to find the slice in the source.
  The full unfiltered 464k-row index is in
  <code>extraction_candidates_index.csv</code> with all six filter
  flags and the rejection reason; the post-filter index is in
  <code>extraction_candidates_filtered.csv</code>.</p>
</section>
""")

# Funnel
funnel_rows = "".join(
    f"<tr><td>{esc(name)}</td><td>{n:,}</td><td>{dropped:,}</td><td>{(n*100/n0):.1f} %</td></tr>"
    for name, n, dropped in funnel
)
sections.append(f"""
<section id='funnel'>
  <h2>2. Quality filter funnel</h2>
  <table class='small'>
    <thead><tr><th>step</th><th>kept</th><th>dropped at this step</th><th>cumulative survival</th></tr></thead>
    <tbody>
      <tr><td>Initial: predicted extraction-worthy by Phase-6 classifier</td><td>{n0:,}</td><td>&mdash;</td><td>100.0 %</td></tr>
      {funnel_rows}
    </tbody>
  </table>
  <p class='hint'>Filter rule keys:
    R1 templated-outline (angle-bracket placeholder in any step) &middot;
    R2 single-cluster repetition (a,a,&hellip;,a) &middot;
    R3 single-scenario (n_distinct_scenarios = 1) &middot;
    R4 overlap-dominated (scen/support &lt; 0.20) &middot;
    R5 mechanism-mismatch (SHL needs &ge;2 orgs) &middot;
    R6 non-closed (length+1 super-pattern with same support and scope &mdash;
       collapses slice variants of the same underlying reuse opportunity).
  </p>
</section>
""")

# Headline (post-filter)
n_total_filt = len(filtered)
n_by_mech = dict(Counter(filtered["mechanism"]))
n_by_scope = dict(Counter(filtered["scope"]))
n_by_L = dict(Counter(filtered["L"]))

mech_table = "".join(
    f"<tr><td>{esc(k)}</td><td>{v:,}</td><td>{v*100/n_total_filt:.1f}</td></tr>"
    for k, v in sorted(n_by_mech.items(), key=lambda kv: -kv[1])
)
scope_table = "".join(
    f"<tr><td>{esc(k)}</td><td>{v:,}</td><td>{v*100/n_total_filt:.1f}</td></tr>"
    for k, v in sorted(n_by_scope.items(), key=lambda kv: -kv[1])
)
L_table = "".join(
    f"<tr><td>{L}</td><td>{c:,}</td><td>{c*100/n_total_filt:.1f}</td></tr>"
    for L, c in sorted(n_by_L.items())
)
repo_table = "".join(
    f"<tr><td class='mono'>{esc(r)}</td><td>{c:,}</td></tr>"
    for r, c in top10_repos
)
sections.append(f"""
<section id='headline'>
  <h2>3. Headline statistics (post-filter)</h2>
  <div class='stats'>
    <div class='stat'><div class='label'>Surviving EW candidates</div><div class='value'>{n_total_filt:,}</div></div>
    <div class='stat'><div class='label'>Distinct repos hosting at least one</div><div class='value'>{n_repos:,}</div></div>
    <div class='stat'><div class='label'>Patterns rendered with full call-site detail</div><div class='value'>{len(target_patterns):,}</div></div>
    <div class='stat'><div class='label'>Slices in raw corpus</div><div class='value'>5,382,249</div></div>
  </div>
  <h3>By mechanism</h3>
  <table class='small'><thead><tr><th>mechanism</th><th>count</th><th>%</th></tr></thead><tbody>{mech_table}</tbody></table>
  <h3>By scope</h3>
  <table class='small'><thead><tr><th>scope</th><th>count</th><th>%</th></tr></thead><tbody>{scope_table}</tbody></table>
  <h3>By slice length L</h3>
  <table class='small'><thead><tr><th>L</th><th>count</th><th>%</th></tr></thead><tbody>{L_table}</tbody></table>
  <p class='hint'>The L=18 bucket is inflated relative to L&lt;18 because the
  closure filter R6 only drops a length-L pattern when a length-(L+1)
  super-pattern exists with the same support, and L_max=18 means no
  L=19 super-patterns exist by construction &mdash; so 100 % of L=18
  candidates pass R6 against 2&ndash;65 % at L=2&ndash;17. The pre-R6
  L distribution decays smoothly through L=18 (see Phase~1 in the
  paper).</p>
  <h3>Top-10 repos by surviving-candidate count</h3>
  <table class='small'><thead><tr><th>repo</th><th>candidates hosted</th></tr></thead><tbody>{repo_table}</tbody></table>
</section>
""")

mech_descriptions = {
    "reusable_scenario":
      ("Reusable-scenario candidates (RQ2: within-repo, cross-file). "
       "Mughal-2024 maps these to the <code>I call feature file:</code> "
       "construct &mdash; lift the slice into its own .feature file and "
       "replace each call site with an invocation."),
    "background":
      ("Background-block candidates (RQ1: within-file, recurring prefix). "
       "Mughal-2024 maps these to a Cucumber <code>Background:</code> block "
       "at the top of the .feature file."),
    "shared_higher_level_step":
      ("Shared higher-level-step candidates (RQ3: cross-organisational, "
       "n_distinct_orgs &ge; 2). Mughal-2024 maps these to the "
       "dynamic-ENUM construct &mdash; an abstracted higher-level step "
       "parameterised by an enum that varies per call site."),
}

idx_for_mech = list(top_per_mech.keys())
for mname, df in top_per_mech.items():
    desc = mech_descriptions[mname]
    pretty = mname.replace('_', ' ')
    sections.append(f"""
<section id='m-{mname}'>
  <h2>4.{idx_for_mech.index(mname)+1}. Top {len(df):,} {pretty} candidates by quality score</h2>
  <p class='hint'>{desc}</p>
  <p class='hint'>Ranking: q = n_distinct_scenarios &times; &radic;support_total
  (&times; n_distinct_orgs for shared_higher_level_step). Click any row to
  expand the representative step text and up to {MAX_SITES_PER_PATTERN}
  call sites &mdash; full instance count is shown in <b>support</b>.</p>
  {render_table(df, f'tab-{mname}')}
</section>
""")

# ----------------------------------------------------------------------
# CSS / JS
# ----------------------------------------------------------------------
CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #fafafa; color: #1a1a1a; line-height: 1.5; }
header { background: #1f2937; color: #f9fafb; padding: 1.2rem 1.6rem; }
header h1 { margin: 0; font-size: 1.4rem; }
header .sub { font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem; }
nav { position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #e5e7eb;
      padding: 0.6rem 1.6rem; z-index: 10; display: flex; gap: 1.2rem; align-items: center; flex-wrap: wrap; }
nav a { color: #2563eb; text-decoration: none; font-size: 0.9rem; }
nav a:hover { text-decoration: underline; }
section { background: #fff; margin: 1.4rem 1.6rem; padding: 1.4rem 1.6rem;
          border: 1px solid #e5e7eb; border-radius: 6px; }
section h2 { margin-top: 0; color: #111827; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.4rem; }
section h3 { color: #374151; margin-top: 1.4rem; }
section h4 { margin: 0.7rem 0 0.3rem; color: #374151; }
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
         gap: 0.8rem; margin: 1rem 0; }
.stat { background: #f3f4f6; padding: 0.8rem 1rem; border-radius: 6px; }
.stat .label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
.stat .value { font-size: 1.4rem; font-weight: 600; color: #111827; margin-top: 0.2rem; }
table.small { width: auto; border-collapse: collapse; margin: 0.6rem 0; font-size: 0.9rem; }
table.small th, table.small td { padding: 0.3rem 0.8rem; border-bottom: 1px solid #e5e7eb; text-align: left; }
table.small th { background: #f9fafb; font-weight: 600; }
table.ranked { width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-size: 0.84rem; }
table.ranked th { background: #f3f4f6; padding: 0.4rem 0.5rem; border-bottom: 2px solid #d1d5db;
                  text-align: left; font-weight: 600; cursor: pointer; user-select: none;
                  position: sticky; top: 50px; }
table.ranked th:hover { background: #e5e7eb; }
table.ranked td { padding: 0.35rem 0.5rem; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
table.ranked tr.row { cursor: pointer; }
table.ranked tr.row:hover { background: #f9fafb; }
table.ranked tr.row.expanded { background: #eff6ff; }
.c-num { text-align: right; font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.c-pat { max-width: 14rem; overflow: hidden; text-overflow: ellipsis; }
.detail-body { padding: 0.7rem 1rem; background: #f9fafb; border-left: 3px solid #2563eb; margin: 0.4rem 0; }
.steps { margin: 0; padding-left: 1.4rem; }
.steps li { margin: 0.15rem 0; }
.sites { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.4rem; }
@media (max-width: 1200px) { .sites { grid-template-columns: 1fr; } }
.site { background: #fff; border: 1px solid #e5e7eb; border-radius: 4px; padding: 0.5rem 0.7rem; }
.site-head { font-size: 0.85rem; word-break: break-all; }
.site-head .repo { font-weight: 600; color: #111827; }
.site-head .file { color: #4b5563; }
.site-head .scenario { color: #6b7280; font-style: italic; }
.site-head .pos { color: #9ca3af; font-size: 0.78rem; margin-left: 0.4rem; }
.site-head .sep { color: #9ca3af; margin: 0 0.25rem; }
.site-steps { margin: 0.3rem 0 0 0; padding-left: 1.6rem; font-size: 0.85rem; color: #1f2937; list-style-type: decimal; }
.no-sites { color: #9ca3af; font-style: italic; }
.cap-note { color: #9ca3af; font-size: 0.85rem; margin-top: 0.4rem; font-style: italic; }
.hint { color: #6b7280; font-size: 0.88rem; }
"""

JS = r"""
document.querySelectorAll('table.ranked').forEach(table => {
  table.addEventListener('click', (e) => {
    const tr = e.target.closest('tr.row');
    if (!tr) return;
    const idx = tr.dataset.idx;
    const detail = table.querySelector(`tr.detail[data-for='${idx}']`);
    if (detail) {
      const open = detail.style.display !== 'none';
      detail.style.display = open ? 'none' : '';
      tr.classList.toggle('expanded', !open);
    }
  });
  table.querySelectorAll('thead th').forEach((th, colIdx) => {
    let asc = true;
    th.addEventListener('click', () => {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr.row'));
      const details = new Map();
      Array.from(tbody.querySelectorAll('tr.detail')).forEach(d =>
        details.set(d.dataset.for, d));
      const num = (s) => { const x = parseFloat(s.replace(/[^\d\.\-]/g, '')); return isNaN(x) ? -Infinity : x; };
      const isNumeric = ![2, 4].includes(colIdx);
      rows.sort((a, b) => {
        const av = a.children[colIdx].textContent.trim();
        const bv = b.children[colIdx].textContent.trim();
        if (isNumeric) return (num(av) - num(bv)) * (asc ? 1 : -1);
        return av.localeCompare(bv) * (asc ? 1 : -1);
      });
      tbody.innerHTML = '';
      rows.forEach(r => {
        tbody.appendChild(r);
        const d = details.get(r.dataset.idx);
        if (d) tbody.appendChild(d);
      });
      asc = !asc;
    });
  });
});
"""

nav_links = [
    ("#primer",                    "What this is"),
    ("#funnel",                    "QA filter funnel"),
    ("#headline",                  "Headline (post-filter)"),
    ("#m-reusable_scenario",       "Reusable scenarios"),
    ("#m-background",              "Background blocks"),
    ("#m-shared_higher_level_step","Shared higher-level"),
]

doc = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Extraction-worthy candidates &mdash; QA-filtered verification report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Extraction-worthy candidates &mdash; QA-filtered verification report</h1>
  <div class='sub'>cukereuse-subscenarios paper-2 &middot; Phase-6/8 classifier output, post-quality-filter &middot; ranked by spread-aware quality score &middot; click any row to drill into a pattern&rsquo;s call-site list</div>
</header>
<nav>{"".join(f"<a href='{href}'>{label}</a>" for href, label in nav_links)}</nav>
{"".join(sections)}
<script>{JS}</script>
</body>
</html>
"""

OUT_HTML.write_text(doc, encoding="utf-8")
size_mb = OUT_HTML.stat().st_size / 1024 / 1024
print(f"\nwrote {OUT_HTML}  ({size_mb:.1f} MB)", flush=True)
print(f"wrote {OUT_CSV_FULL}  (full 464k unfiltered index with flags)", flush=True)
print(f"wrote {OUT_CSV_FILT}  (post-filter index)", flush=True)
