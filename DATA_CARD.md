# Datasheet for cukereuse-subscenarios

Follows the template from Gebru et al., *Datasheets for Datasets*, CACM 64(12), 2021.

**Version:** 0.1.0 (May 2026). Corresponds to the repository commit at release.

---

## 1. Motivation

- **For what purpose was the dataset created?** To support reproduction and extension of paper 3 of the cukereuse arc: paraphrase-robust *subsequence-level* mining of recurring step subsequences ("slices") in public Cucumber/Gherkin test suites, with three-author labels and classifier baselines for an *extraction-worthy* judgement and a downstream three-way *mechanism* judgement aligned to the three reuse patterns of Mughal (2024).
- **Who created the dataset and who funds it?** Ali Hassaan Mughal (Texas Wesleyan University), Noor Fatima (National University of Sciences and Technology, Pakistan), and Muhammad Bilal (Technical University of Munich). No external funding.
- **Any other comments?** The dataset is a *derived analytical bundle* keyed to the cukereuse v0.1.0 step corpus ([cukereuse-release](https://github.com/amughalbscs16/cukereuse-release)). It does not redistribute raw `.feature`-file bodies; those are pointer-only via the cukereuse pointer plus `rehydrate.py`.

## 2. Composition

- **What do the instances represent?** Three primary record types:
  1. **Slice records** in `analysis/slices.parquet` — one row per contiguous L-step window in a named non-Background scenario, with `L ∈ [2, 18]`.
  2. **Pattern records** in `analysis/exact_subsequence_ranking.parquet` — one row per distinct sequence of cukereuse hybrid cluster ids, with global recurrence counts under three scopes (within-file, within-repo, cross-organisational).
  3. **Labelled records** in `methodology/labels.jsonl` — 200 stratified slices labelled by three authors against the written rubric in `methodology/LABELING_RUBRIC_paper2_slice_level.md`.
- **How many instances are there?**
  - 5,382,249 slices over 126,621 named non-Background scenarios with length ≥ 2.
  - 692,020 distinct cluster-id patterns (support ≥ 1); 595,857 with support ≥ 2 (input to extraction-worthy ranking); 464,073 predicted-yes patterns (mechanism-classified).
  - 200 stratified labelled slices, 60 of which are in a three-way overlap subset.
- **What does each instance contain?**
  - **Slice (`slices.parquet`):** `repo`, `file_path`, `scenario`, `line_start`, `line_end`, `L`, `cluster_id_seq` (list of cukereuse cluster ids), `text_seq` (canonical step text per position).
  - **Pattern (`exact_subsequence_ranking.parquet`):** `pattern` (comma-joined cluster ids), `L`, `support_total`, `n_distinct_files`, `n_distinct_repos`, `n_distinct_orgs`, `max_within_file_recurrence`, `max_within_repo_files`, `outlier_fraction`, `scope` ∈ {RQ1, RQ2, RQ3}, `stratum`, `canonical_text_seq`.
  - **Labelled record (`labels.jsonl`):** all pattern fields plus `assignment` ∈ {author_A, author_B, author_C}, `labels_per_author` (one entry per assigned labeller with `extraction_worthy` ∈ {yes, no, uncertain, flagged-spec}, `mechanism` ∈ {n/a, background, reusable_scenario, shared_higher_level_step}, free-text `notes`), aggregated `label_extraction_worthy` and `label_mechanism`.
- **Is there a label or target?** `methodology/labels.jsonl` carries two labels per slice: an extraction-worthy 4-category label (yes / no / uncertain / flagged-spec) and a mechanism 5-category label (n/a / background / reusable_scenario / shared_higher_level_step / unsure). Both labels were applied directly by the named authors against the rubric. Every label record carries the `notes` clause that fired so rubric application is auditable per slice without re-annotation.
- **Are there recommended data splits?** No built-in train/test split for the labelled set; the classifier scripts use 5-fold stratified cross-validation on the 200 labelled slices and report out-of-fold metrics with bootstrap CIs.
- **Are there errors, noise, or redundancies?**
  - Recurrence is the subject of study; high duplication of slice patterns is expected and is the central empirical signal.
  - **Auto-generated specs.** A small number of repositories ship machine-generated `.feature` suites whose recurrence counts are generator artefacts, not maintenance burden. These are flagged via `analysis/spec_suite_outliers.csv` and excluded from headline rates; in the labelled pool such slices receive `extraction_worthy = flagged-spec`.
  - **Scenario Outline duplication.** Outline rows expand to multiple step instances at runtime but reflect a single authoring decision; slices whose support exactly matches the example-table size of an outline are flagged similarly.
- **Sensitive content?** No PII is retained. All upstream metadata is derived from the cukereuse v0.1.0 step corpus, which is itself a pointer-based release of public GitHub repositories. Author identity in `labels.jsonl` is reduced to A/B/C tags; the mapping (A = Mughal, B = Fatima, C = Bilal) is documented here and in the labelling rubric.

### Coarse breakdown

| Axis | Count |
|---|---:|
| repositories with at least one slice | 339 |
| distinct upstream owners (segment before first underscore in repo slug; a mix of Organisation and User accounts on GitHub) | 276 |
| `.feature` files with at least one slice | 21,946 |
| named non-Background scenarios with length ≥ 2 | 134,635 |
| ditto, after cluster-id filter (≥ 2 clustered steps) | 126,621 |
| total steps in upstream cukereuse corpus | 1,113,616 |
| slices (L ∈ [2, 18]) | 5,382,249 |
| distinct cluster-id patterns | 692,020 |
| ditto, support ≥ 2 | 595,857 |
| predicted-yes (extraction-worthy) patterns | 464,073 |
| labelled slices (stratified pool) | 200 |
| labelled slices (3-way overlap subset) | 60 |
| LLM-judge models | 2 (`openai/gpt-oss-120b`, `inclusionai/ling-2.6-1t`) |

### Per-author labelling distribution

| Labeller | Pool slices labelled | (yes / no / uncertain / flagged-spec) on assigned slices |
|---|---:|---|
| A — Mughal | 106 | 77 / 12 / 4 / 13 |
| B — Fatima | 106 | 72 / 11 / 11 / 12 |
| C — Bilal | 108 | 72 / 23 / 0 / 13 |

The 60-slice three-way overlap subset is double-counted across authors above; the union covers exactly 200 distinct slices.

### Inter-rater agreement (60-slice three-way overlap)

| Label | Categories | Fleiss' κ |
|---|---|---:|
| extraction_worthy | 4 (yes / no / uncertain / flagged-spec) | 0.560 |
| mechanism | 5 (n/a / background / reusable_scenario / shared_higher_level_step / unsure) | 0.788 |

Pairwise raw agreement (extraction-worthy / mechanism): A-B 0.717 / 0.800; A-C 0.850 / 0.933; B-C 0.750 / 0.800.

## 3. Collection process

The pipeline runs in eleven scripted stages in `scripts/`. Every stage reads its inputs from `analysis/` (or `methodology/`) and writes its outputs back into the same directory, so the bundle is self-contained.

1. **Slice extraction (`01_extract_slices.py`).** For each named non-Background scenario in the cukereuse corpus with at least two consecutive cluster-id-bearing steps, enumerate every contiguous window of length L = 2..18. Output: `slices.parquet`.
2. **Subsequence counting (`02_count_subsequences.py`).** Group by `cluster_id_seq` and accumulate three-scope recurrence counts. Output: an unfiltered ranking parquet.
3. **Outlier refinement and scope tagging (`02c_refine_ranking_and_outliers.py`).** Compute `outlier_fraction` (max-file / total-support) and assign each pattern to RQ1 / RQ2 / RQ3 by where its recurrence concentrates. Output: `exact_subsequence_ranking.parquet`.
4. **Slice embedding clustering (`04_slice_embedding_clustering.py`).** SBERT-encode the `text_seq` of each slice, project to 2D via UMAP, cluster with HDBSCAN. Outputs: `slice_embeddings.npz` (820 MB, **not redistributed** — regenerable), `cluster_embeddings.npz` (89 MB, **not redistributed**), `slice_clusters.parquet` (cluster id + density score per slice, redistributed).
5. **Labelling pool construction (`05a_build_labeling_pool.py`).** Stratified sample of 200 slices over (L bucket × scope × support bucket) plus a deterministic 60-slice three-way overlap subset. Output: `methodology/labeling_pool.jsonl`.
6. **Three-author labelling (manual).** Each author labelled their assigned slices against the written rubric. The aggregated output, with one row per slice and a `labels_per_author` field with the per-author labels and rubric-clause notes, is `methodology/labels.jsonl`.
7. **Aggregation and Fleiss' κ (`06_aggregate_labels_and_kappa.py`).** Computes 4-category and 5-category Fleiss' κ on the overlap subset and per-author distributions. Output: `methodology/labels_summary.json`.
8. **Extraction-worthy classifier (`06b_train_extraction_classifier.py`).** XGBoost on the 200 labelled slices with 5-fold stratified CV and 1000-bootstrap-iteration confidence intervals. Output: `analysis/extraction_classifier_metrics.json`, `extraction_classifier_predictions.parquet`, `extraction_classifier_feature_importance.json`.
9. **LLM-judge runs (`07_llm_judge.py`, `07b_llm_judge_agreement.py`).** Two open-weight judges via OpenRouter; per-slice JSON outputs in `methodology/llm_judge/`; agreement metrics vs. author majority in `methodology/llm_judge_agreement.json`.
10. **Mechanism classifier (`08_train_mechanism_classifier.py`).** XGBoost on the 143 yes-only slices with 5-fold stratified CV. Output: `mechanism_classifier_metrics.json`, `mechanism_predictions.parquet`.
11. **Population rollups (`09a_preliminary_rollups.py`, `09b_post_classifier_headline.py`, `11_extraction_candidates_html_report.py`).** Project the classifier onto the 595,857-pattern population and produce headline numbers + a self-contained HTML report.

**Over what timeframe?** Pipeline runs were completed between 2026-04-19 (upstream cukereuse mining) and 2026-05-09 (this release). The slice inventory is deterministic given the cukereuse v0.1.0 cluster ids; LLM-judge runs are seedless and may differ if re-run.

**Ethical review?** Data is drawn exclusively from public GitHub repositories that survived the cukereuse v0.1.0 stars/archival filter. No deanonymisation. Author labels are from the three named authors only; no human-subjects programme was conducted (in particular, no external raters were recruited, and no LLM was used to produce the labels — LLMs appear only as a separately-evaluated judge baseline whose outputs live under `methodology/llm_judge/`).

## 4. Preprocessing, cleaning, labelling

- **Cluster-id source.** Every step's cluster id comes from the cukereuse v0.1.0 hybrid clusterer (cosine + Levenshtein band; see paper 2). Steps that fail to cluster (singleton or filtered) are dropped before slice extraction; this drops 8,014 of 134,635 named scenarios (5.95%) and 82,162 of 1,113,616 steps (7.4%).
- **Slice boundaries.** Slices are intra-scenario only; no slice spans across scenario boundaries or across feature files. Background blocks are excluded (the upstream cukereuse corpus already separates them).
- **Outlier fraction.** Defined as `max_within_file_recurrence / support_total`. Patterns with `outlier_fraction ≥ 0.3` are downweighted in the rule baseline because they signal in-file repetition rather than reusable cross-context behaviour.
- **Labelling.** Manual, three-author against the rubric in `methodology/LABELING_RUBRIC_paper2_slice_level.md`. Authors labelled independently before `06_aggregate_labels_and_kappa.py` merged the per-author outputs. The 60-slice overlap subset was labelled by all three authors before the disjoint 140-slice tail was assigned.

## 5. Uses

Supported tasks for v0.1.0:

- Reproduction of the paper-3 headline numbers (extraction-worthy rate, mechanism mix, RQ1/RQ2/RQ3 population shares).
- Comparison of new classifier baselines against the three-author labels with a fixed 5-fold split.
- Comparison of new LLM judges against the human authors using the included judge protocol; raw outputs of two open-weight judges are included for direct comparison.
- Re-running the pipeline against newer cukereuse corpus snapshots (recreate `slices.parquet` from a new `clusters_hybrid.parquet`).

Out-of-scope tasks:

- Step-level duplicate-detection benchmarking — use the [cukereuse](https://github.com/amughalbscs16/cukereuse-release) release for that.
- Adoption / acceptance studies — this release captures the *discovery* layer, not the maintainer-acceptance layer.

Restrictions:

- Do not use the slice pointers to violate upstream repositories' Terms of Service (rate limits, scraping). Rehydration is supported via the upstream cukereuse `rehydrate.py`.
- Do not redistribute LLM-judge raw outputs as if they were author labels; the two are tagged separately.

## 6. Distribution

- **Release channel:** GitHub repository at [amughalbscs16/cukereuse_subscenarios_release](https://github.com/amughalbscs16/cukereuse_subscenarios_release). Apache-2.0 for source code and analytical schema.
- **Bundle size:** approximately 154 MB total (153 MB analytical artefacts + 1 MB methodology).
- **Heavy artefacts not redistributed:** `slice_embeddings.npz` (820 MB) and `cluster_embeddings.npz` (89 MB). Both are regenerable via `scripts/04_slice_embedding_clustering.py` from `slices.parquet`.

## 7. Maintenance

- **Maintainers:** the three authors.
- **Errata:** filed as GitHub issues on [amughalbscs16/cukereuse_subscenarios_release](https://github.com/amughalbscs16/cukereuse_subscenarios_release).
- **Versioning:** a frozen snapshot corresponds to each paper version; major artefact updates will be tagged as GitHub releases (`v0.1.0`, `v0.2.0`, …).
- **Upstream dependency:** this release pins to the cukereuse v0.1.0 step corpus. If that release re-versions, this repository will publish a corresponding new tag.
