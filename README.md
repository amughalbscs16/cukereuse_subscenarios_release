# cukereuse-subscenarios: subsequence-level refactoring candidate mining for Cucumber/Gherkin

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research_release-brightgreen.svg)](CITATION.cff)
[![Companion: cukereuse](https://img.shields.io/badge/companion-cukereuse--release-informational.svg)](https://github.com/amughalbscs16/cukereuse-release)

Reproduction package for the paper

> Mughal, A. H., Fatima, N., & Bilal, M. (2026). *Mining Subscenario Refactoring Opportunities in Behaviour-Driven Test Suites: A Paraphrase-Robust, Three-Author-Labelled Empirical Study with Classifier and LLM-Judge Baselines (339 Repositories).* Manuscript under review at *Software Quality Journal* (Springer).

This is paper 3 of a three-paper arc on test reuse in Cucumber-style BDD. Paper 1 (Mughal 2024) introduced three concrete reuse mechanisms in Cucumber-Java; paper 2 ([cukereuse](https://github.com/amughalbscs16/cukereuse-release)) measured *step-level* duplication on a 1.1M-step open corpus; paper 3 (this repository) lifts the analysis from steps to *contiguous step subsequences* (slices), ranks them by refactoring suitability, and maps each surviving candidate to one of paper 1's three mechanisms.

## What's in the box

A reproducible bundle covering the full paper-3 pipeline:

- **Slice inventory** — every contiguous L-step window (L = 2..18) drawn from 126,621 named, non-Background scenarios across **339 repositories / 276 organisations**, keyed by paraphrase-robust cluster identifiers from the cukereuse hybrid clusterer.
- **Subsequence ranking** — exact-pattern recurrence counts under three scopes (within-file / within-repo / cross-organisational), with outlier de-noising and filtered candidate exports.
- **Labelled pool** — 200 stratified slices (5 length × 3 scope × 4 support strata) labelled by all three authors against a written rubric, with a 60-slice overlap subset for inter-rater agreement (Fleiss' κ = 0.560 on the 4-category extraction-worthy label, 0.788 on the 5-category mechanism label).
- **Classifier baselines** — XGBoost extraction-worthy classifier (5-fold OOF F₁ = 0.891, AUC = 0.879) and mechanism classifier (5-fold OOF accuracy = 0.965, macro-F₁ = 0.955); rule-baseline comparison via McNemar's test (χ² = 5.69, p = 0.017).
- **LLM-judge replication** — two open-weight judges (`openai/gpt-oss-120b`, `inclusionai/ling-2.6-1t`) scored on the same 200-slice pool; agreement metrics, Cohen's κ vs. authors, and pairwise McNemar tests against the XGBoost classifier.
- **Population rollups** — extraction-worthy headline numbers projected onto the 126,621 named-scenario population: 75.0% RQ1-eligible, 59.5% RQ2-eligible, 11.7% RQ3-eligible at the scenario level; 83.2% / 43.7% of repositories carry at least one RQ2 / RQ3 candidate.

## Who this is for

- **Maintainers of large BDD suites** assessing how much of their suite is recurrent enough to refactor into Backgrounds, reusable scenarios, or shared higher-level steps.
- **Test-tooling researchers** building on a reproducible corpus of recurring Gherkin slices with explicit three-author labels and rubric-firing rules per slice.
- **LLM-as-judge researchers** comparing open-weight model agreement against a fixed, human-labelled, rubric-anchored test-engineering judgement task.

## Install

```bash
git clone https://github.com/amughalbscs16/cukereuse_subscenarios_release.git
cd cukereuse_subscenarios_release
uv sync                                  # or: pip install -e .
```

Requires Python 3.10+. Torch is pulled from the CPU index; no GPU is needed for the analytical scripts (embedding regeneration is faster on GPU but optional — the slice clusters are shipped as a parquet).

## What's released

```
cukereuse_subscenarios_release/
├── analysis/                 (153 MB derived analytical artefacts)
├── methodology/              (labels, rubric, LLM-judge raw outputs)
├── scripts/                  (reproduction pipeline, 21 .py files)
├── DATA_CARD.md              (per-artefact provenance and schema)
├── CITATION.cff
├── LICENSE                   (Apache-2.0)
├── NOTICE
├── pyproject.toml
└── README.md
```

### `analysis/` — derived artefacts

| File | Rows | Content |
|------|-----:|---------|
| `slices.parquet` | 5,382,249 | every (scenario, L) slice with `cluster_id_seq`, `repo`, `file`, `line_start`, `line_end`, scope counts |
| `slice_clusters.parquet` | 5,382,249 | UMAP+HDBSCAN cluster id + density score per slice (paraphrase-robust slice identity) |
| `exact_subsequence_ranking.parquet` | 692,020 | distinct cluster-id patterns with `support_total`, `n_distinct_repos`, `n_distinct_files`, `outlier_fraction`, scope label |
| `extraction_classifier_predictions.parquet` | 595,857 | XGBoost extraction-worthy probabilities (the support ≥ 2 subset of the ranking) |
| `mechanism_predictions.parquet` | 464,073 | predicted mechanism (`background` / `reusable_scenario` / `shared_higher_level_step`) for each predicted-yes pattern |
| `extraction_candidates_index.csv` | 464,073 | the predicted-yes patterns with full per-candidate metadata, ready for a maintainer-facing report |
| `extraction_candidates_filtered.csv` | 84,564 | curated subset used for the inspection-burden estimate (after outlier-fraction and support filters) |
| `reusable_scenario_report.html` | — | self-contained HTML report of the top reusable-scenario candidates |
| `corpus_summary.json` | — | one-stop corpus shape (339 repos, 276 orgs, 126,621 scenarios, 1,113,616 steps) |
| `rule_baseline_metrics.json` | — | rule baseline + XGBoost OOF + McNemar (vs. R1, vs. each LLM judge) |
| `extraction_classifier_metrics.json` | — | per-fold and bootstrap-CI metrics for XGBoost extraction-worthy classifier |
| `mechanism_classifier_metrics.json` | — | per-fold and bootstrap-CI metrics for XGBoost mechanism classifier |
| `post_classifier_headline.json` | — | scenario- and repo-level extraction-worthy headline rates (RQ1/RQ2/RQ3) |
| `preliminary_rollups.json` | — | preliminary headline numbers (pre-classifier sanity rollup) |
| `phase2c_summary.json` | — | exact-subsequence count summary (counts of distinct patterns by scope) |
| `slices_summary.json` | — | slice inventory summary (per-L bucket counts) |
| `slice_clusters_summary.json` | — | cluster-density summary |
| `exact_subsequence_summary.json` | — | recurring-pattern summary |
| `extraction_classifier_feature_importance.json` | — | XGBoost gain-based importances |
| `scenario_identity_audit.json` | — | scenario-name uniqueness audit |
| `scenario_length_distribution.json` | — | distribution of scenario lengths (steps per named scenario) |
| `scenario_length_histogram.csv` | — | tabular scenario-length histogram |
| `spec_suite_outliers.csv` | — | repos flagged as auto-generated spec suites (excluded from headline rates) |

What is **not** redistributed: `slice_embeddings.npz` (820 MB) and `cluster_embeddings.npz` (89 MB). Both are regenerable via `scripts/04_slice_embedding_clustering.py` from `slices.parquet`. The shipped `slice_clusters.parquet` already carries the cluster ids those embeddings produced, so most reproduction work does not need them.

### `methodology/` — labels and judges

| File | Content |
|------|---------|
| `LABELING_RUBRIC_paper2_slice_level.md` | 7-section written rubric (yes / no / uncertain / flagged-spec; mechanism = background / reusable_scenario / shared_higher_level_step / unsure) |
| `labels.jsonl` | 200 author-labelled slices, one per line, with `labeler` ∈ {A, B, C}, the rubric clause that fired, free-text notes |
| `labeling_pool.jsonl` | the stratified 200-slice pool that was sent for labelling |
| `labeling_pool_stats.json` | per-stratum counts of the labelling pool |
| `labels_summary.json` | 4-category and 5-category Fleiss' κ on the 60-slice overlap subset, per-author distributions |
| `llm_judge/openai_gpt-oss-120b_free.jsonl` | per-slice JSON output from `openai/gpt-oss-120b` via OpenRouter |
| `llm_judge/inclusionai_ling-2.6-1t_free.jsonl` | per-slice JSON output from `inclusionai/ling-2.6-1t` via OpenRouter |
| `llm_judge_agreement.json` | Cohen's κ, accuracy, F₁ per LLM judge against author majority; inter-LLM Fleiss' κ |

The labels are author-produced. They are not LLM-generated; LLMs appear in this study only as a separately-evaluated judge baseline whose outputs are tagged in `methodology/llm_judge/`. See the rubric file for the per-clause decision criteria the authors applied.

### `scripts/` — reproduction pipeline

Numbered to match the pipeline order described in the paper. Each script reads from `analysis/` and writes back into `analysis/` (or `methodology/`).

```
00_scenario_length_distribution.py     scenario shape audit
00b_scenario_count_audit.py            named-scenario count sanity
01_extract_slices.py                   build slices.parquet
01a_probe_cluster_schema.py            sanity-probe cukereuse cluster ids
02_count_subsequences.py               build exact_subsequence_ranking.parquet
02b_inspect_top_patterns.py            human-readable top-pattern dump
02c_refine_ranking_and_outliers.py     outlier filter + scope tagging
04_slice_embedding_clustering.py       SBERT + UMAP + HDBSCAN slice clusters
05a_build_labeling_pool.py             stratified 200-slice pool sampler
05b_pilot_label_sample.py              warm-up sample for rubric pilot
06_aggregate_labels_and_kappa.py       merge per-author labels + Fleiss' κ
06b_train_extraction_classifier.py     XGBoost extraction-worthy classifier
07_llm_judge.py                        OpenRouter LLM-judge driver
07b_llm_judge_agreement.py             κ + F₁ + McNemar vs authors
08_train_mechanism_classifier.py       XGBoost mechanism classifier
09a_preliminary_rollups.py             pre-classifier headline rollup
09b_post_classifier_headline.py        post-classifier RQ1/RQ2/RQ3 rollup
11_extraction_candidates_html_report.py self-contained HTML report
_parquet_loader.py                     shared parquet helpers
plot_paper_figures.py / _v2.py         figure regeneration
```

To re-run the pipeline end-to-end (the upstream cukereuse v0.1.0 corpus must be present in a sibling directory; see `DATA_CARD.md` for the expected layout):

```bash
uv run python scripts/01_extract_slices.py
uv run python scripts/02_count_subsequences.py
uv run python scripts/02c_refine_ranking_and_outliers.py
uv run python scripts/04_slice_embedding_clustering.py     # ~820MB embeddings
uv run python scripts/05a_build_labeling_pool.py
# author-labelling step is manual; outputs land in methodology/labels.jsonl
uv run python scripts/06_aggregate_labels_and_kappa.py
uv run python scripts/06b_train_extraction_classifier.py
uv run python scripts/07_llm_judge.py                      # needs OPENROUTER_API_KEY
uv run python scripts/07b_llm_judge_agreement.py
uv run python scripts/08_train_mechanism_classifier.py
uv run python scripts/09b_post_classifier_headline.py
uv run python scripts/11_extraction_candidates_html_report.py
```

## Key numbers

These are the citable headline numbers from this release. The accompanying paper presents them with full methodology, bootstrap confidence intervals, and the licence-stratification, classifier-baseline, and LLM-judge comparisons.

- **Slice inventory:** 339 repositories / 276 organisations, 21,946 `.feature` files, 126,621 named non-Background scenarios with length ≥ 2, 5,382,249 slices over L ∈ [2, 18].
- **Distinct cluster-id patterns:** 692,020 total; 595,857 with support ≥ 2 (the input to extraction-worthy ranking).
- **Author-labelled extraction-worthy rate (200-slice stratified pool):** 71.7% yes, 9.1% flagged-spec, 19.2% no/uncertain.
- **Inter-rater agreement (60-slice overlap, three authors):** Fleiss' κ = 0.560 (4-category extraction-worthy), Fleiss' κ = 0.788 (5-category mechanism).
- **XGBoost extraction-worthy classifier:** 5-fold OOF F₁ = 0.891, AUC = 0.879. Bootstrap-95% CI on F₁: [0.852, 0.927].
- **Rule baseline (R1: outlier_fraction < 0.3):** F₁ = 0.836. McNemar XGB-vs-R1: χ² = 5.69, p = 0.017.
- **XGBoost mechanism classifier (3-class):** 5-fold OOF accuracy = 0.965, macro-F₁ = 0.955.
- **Best LLM-judge (`openai/gpt-oss-120b`):** binary Cohen's κ = 0.348, F₁(yes) = 0.728. McNemar XGB-vs-LLM: χ² = 14.42, p = 1.5×10⁻⁴.
- **Population headlines (post-classifier, scenario level over 126,621 scenarios):**
  RQ1-eligible (within-file Background candidate): 75.0%;
  RQ2-eligible (within-repo reusable scenario): 59.5%;
  RQ3-eligible (cross-organisational shared step): 11.7%.
- **Population headlines (repo level over 339 repositories):**
  ≥ 1 RQ2 candidate: 83.2%; ≥ 1 RQ3 candidate: 43.7%.

## Companion paper-2 release

The 1.1M-step Gherkin corpus, the per-step labelled pairs, and the cukereuse hybrid clusterer that produces the cluster ids underlying every slice in this release live in [amughalbscs16/cukereuse-release](https://github.com/amughalbscs16/cukereuse-release). That repository is the upstream dependency: scripts in this release expect cukereuse cluster identifiers to be available for every step. The two repositories together fully cover the paper-2 (step-level duplicate detection) and paper-3 (slice-level refactoring-candidate mining) reproduction.

## Citation

Please cite both this software/data archive and the preprint. The `CITATION.cff` file is rendered by GitHub's "Cite this repository" widget; its `preferred-citation` field points to the paper.

### BibTeX

```bibtex
@misc{mughal2026cukereusesubscenarios,
  title        = {Mining Subscenario Refactoring Opportunities in Behaviour-Driven Test Suites: A Paraphrase-Robust, Three-Author-Labelled Empirical Study with Classifier and LLM-Judge Baselines (339 Repositories)},
  author       = {Ali Hassaan Mughal and Noor Fatima and Muhammad Bilal},
  year         = {2026},
  note         = {Manuscript under review at \emph{Software Quality Journal} (Springer)}
}

@software{mughal2026cukereusesubscenariosv010,
  title     = {cukereuse-subscenarios v0.1.0: subsequence-level refactoring candidate mining for Cucumber/Gherkin},
  author    = {Mughal, Ali Hassaan and Fatima, Noor and Bilal, Muhammad},
  year      = {2026},
  version   = {0.1.0},
  publisher = {GitHub},
  url       = {https://github.com/amughalbscs16/cukereuse_subscenarios_release}
}
```

## Licence

Apache-2.0 for the source code, derived analytical schema, and the labelling rubric. See [LICENSE](LICENSE).

The slice inventory carries upstream `repo` and `commit_sha` pointers; raw `.feature` file bodies are not redistributed and remain under their source-repository licences. The corresponding pointer-based release is [cukereuse-release](https://github.com/amughalbscs16/cukereuse-release), whose `rehydrate.py` fetches the original feature files on demand.

## Authors

- **Ali Hassaan Mughal**, Independent Researcher, Applied MBA (Data Analytics), Texas Wesleyan University. ORCID [0000-0002-0724-9197](https://orcid.org/0000-0002-0724-9197). `alihassaanmughal.work@gmail.com`.
- **Noor Fatima**, Independent Researcher, B.E. Computer Engineering, National University of Sciences and Technology (NUST), Pakistan. `nfatima.bce25seecs@seecs.edu.pk`.
- **Muhammad Bilal**, Independent Researcher, M.Sc. Management, Technical University of Munich. ORCID [0000-0003-4106-0256](https://orcid.org/0000-0003-4106-0256). `m.bilal@tum.de`.
