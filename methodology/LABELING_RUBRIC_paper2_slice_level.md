# Paper 2 — slice-level labelling rubric (DRAFT v0.1)

> **Status:** draft. Not yet validated by inter-rater agreement.
> Three authors (Mughal, Fatima, Bilal) will pilot on a 20-slice
> warm-up set, refine the wording, then label the full N=200
> stratified sample with a 60-slice overlap subset for Fleiss' kappa.
> Methodology mirrors paper 1 (cukereuse) and extends it from pair
> level to slice level.

---

## 1. Scope and unit of analysis

A **slice** is a contiguous run of *L* consecutive Gherkin steps
(`2 <= L <= 18`) inside a single scenario, drawn from the cukereuse
corpus (347 repos, 136,970 scenarios, 1.1M steps). Each slice is
identified by:

- a sequence of cukereuse hybrid `cluster_id`s (paraphrase-robust step identity),
- the canonical text of each step (for human reading),
- a representative `(repo_slug, file_path, scenario)` triple.

The slice is presented to the labeller along with three scope signals:

- **`max_within_file_recurrence`** — max # of distinct scenarios in any
  single `.feature` file containing this slice. Signal for **RQ1
  (Background candidate)**.
- **`max_within_repo_files`** — max # of distinct files in any single
  repo containing this slice. Signal for **RQ2 (reusable scenario
  candidate)**.
- **`n_distinct_repos`** — # of distinct repos containing this slice.
  Signal for **RQ3 (shared higher-level step candidate)**.

---

## 2. Labels

Each slice receives **two** independent labels:

### 2.1 Label A — extraction-worthy

A binary judgement: **is this slice a good candidate for extraction
into a Mughal-2024 reuse mechanism?** Yes / no / uncertain.

**Yes** — the slice represents a coherent, reusable unit of
behaviour that a maintainer would benefit from extracting. Specifically,
all of the following must hold:

- **B-1. Coherent.** The slice represents one logically grouped piece
  of behaviour (a setup, a precondition check, a domain action, a
  reusable assertion block). A reader unfamiliar with the scenario
  could read the slice and name what it does.
- **B-2. Stable.** The slice is unlikely to need scenario-specific
  parameterisation that varies between callers. (If parameterisation
  is needed, the parameters are conventional placeholders like quoted
  values — these are still extractable.)
- **B-3. Non-trivial.** The slice carries meaningful behaviour. A
  slice consisting of only `Given I am logged in` (a single step
  promoted to L=2 by a duplicated Given) is borderline; flag rather
  than mark yes.
- **B-4. Stand-alone.** The slice has a clear start (typically a Given
  or a When in a setup chunk) and end (typically a Then or the natural
  finish of an action). It does not begin or end mid-thought.
- **B-5. Recurrence is genuine.** The recurrence count is not a side
  effect of generated specifications, parametrised tests, or
  scenario-outline duplication. Generated suites get **flagged-spec**
  rather than yes (see &sect;3).

**No** — at least one of the following:

- **N-1. Incoherent slice boundary.** The slice cuts a When-Then
  binding, splits a multi-step assertion, or otherwise represents a
  syntactic accident rather than a behavioural unit.
- **N-2. Domain-specific noise.** The recurrence is caused by repeated
  use of a specific value or entity that wouldn't generalise (e.g.,
  many scenarios checking `Then I see "OK"` does not warrant extraction).
- **N-3. Already-extracted.** The slice is already a Background block
  in another file or already a `I call feature file` in this repo;
  re-extraction is redundant.
- **N-4. Slice too short for value.** L=2 with low semantic content
  ("Given X" + "When Y" with no shared object). Extraction would not
  reduce maintenance burden.
- **N-5. Generated / outline duplication.** The recurrence is the
  product of `Scenario Outline` example expansion or generated test
  suites; the &ldquo;real&rdquo; scenario-author wrote one outline.
  See &sect;3.

**Uncertain** — labeller cannot decide between yes and no after
careful reading. Used sparingly; counts toward Fleiss-&kappa; agreement
calculation.

### 2.2 Label B — recommended Mughal-2024 mechanism

Conditional on Label A = yes, choose one:

- **`background`** — extract to a Background block at the top of the
  same `.feature` file. Apply when the slice recurs in &ge; 2
  scenarios within one file and is currently duplicated. Mughal 2024
  notes that Background does not solve cross-file recurrence; only
  use this for single-file Background candidates.

- **`reusable_scenario`** — extract to a new file at
  `features/reusable/<group>/<name>.feature`; insert
  `And I call feature file <ENUM_CONST>` at each call site; regenerate
  ENUMs via Mughal 2024 Algorithm 1. Apply when the slice recurs
  across &ge; 2 different files in the same repo.

- **`shared_higher_level_step`** — extract to a custom higher-level
  step backed by Mughal 2024 Algorithm 2. Apply when the slice recurs
  across &ge; 2 distinct repos. The slice becomes a single named step
  whose definition wraps the underlying sequence.

- **`unsure`** — slice is extraction-worthy (Label A = yes) but no
  single mechanism fits cleanly. Recorded; useful for post-hoc rubric
  refinement.

If Label A = no or uncertain, Label B is left as `n/a`.

---

## 3. Edge case: spec-suites and outlines

The cukereuse corpus contains two pattern-types that inflate
recurrence counts but should NOT be marked extraction-worthy:

- **Generated specifications.** Examples observed in pre-flight: the
  `local-web-services/local-web-services` repo contains AWS service
  specs (memorydb, neptune) generated from a service definition. A
  slice can recur 1,740 times inside one file, but the &ldquo;recurrence&rdquo;
  is a generator artefact, not a copy-paste maintenance burden.
- **Scenario outlines.** A `Scenario Outline` with N example rows
  produces N step-instances at runtime but is authored once. Any
  recurrence whose support is exactly the example-table size of an
  outline reflects authoring intent, not duplication.

For both, label as **`flagged-spec`** rather than yes/no:

- Detection cue for generated specs: support_total / max_within_file_recurrence > 5
  AND the canonical text shows obvious string-template structure
  (`{placeholder}`, `<name>`, repeated entity names).
- Detection cue for outlines: scenario name has `[N]` numbering or
  scenario position-within-file is monotonic.

Slices labelled `flagged-spec` are excluded from headline percentages
but reported separately as a category in the methods section.

---

## 4. Sampling strategy

- **N = 200 slices** for the main labelling pool.
- **Stratification** along three axes (preserves coverage):
  - **L bucket:** {2,3}, {4,5}, {6,7,8}, {9..12}, {13..18} &rarr; 5 strata
  - **Scope:** within-file (RQ1 candidate), within-repo (RQ2), cross-repo (RQ3) &rarr; 3 strata
  - **Support bucket:** {2..5}, {6..20}, {21..100}, {>100} &rarr; 4 strata
- Total cells: 5 &times; 3 &times; 4 = 60. We sample &le; 4 per cell, with
  proportional re-balance for empty cells.
- **Overlap subset:** 60 slices labelled by all three authors for
  Fleiss-&kappa;; the remaining 140 split 70/35/35 across authors.

---

## 5. Worked examples

### Example 1 (yes / reusable_scenario)

`cluster_id_seq = [140, 43]`, L = 2, support = 621, max_within_repo_files = 620.
Texts:
- `When I run "git-town undo"`
- `Then Git Town runs the commands`

**Label A:** yes &mdash; coherent (undo + verification), stable, non-trivial,
stand-alone, genuine recurrence (621 files in git-town).
**Label B:** `reusable_scenario` &mdash; recurs across many files in one repo;
the natural extraction is a `features/reusable/git_town/undo_check.feature`
called as `I call feature file GIT_TOWN_UNDO_CHECK`.

### Example 2 (yes / shared_higher_level_step)

`cluster_id_seq = [1, 0]`, L = 2, support = 11460, n_distinct_repos = 5.
Texts:
- `Given the request is sent`
- `Then the response status is 200 OK`

**Label A:** yes &mdash; coherent (API call + success assertion), stable
(no parameterisation needed), non-trivial, stand-alone, genuine recurrence
across distinct organisations (DataDog, NHS, etc.).
**Label B:** `shared_higher_level_step` &mdash; cross-repo. Mughal 2024
Algorithm 2 emits a step-definition method like
`@And("I successfully send the request and assert 200")` whose body wraps
the underlying steps.

### Example 3 (no / N-2 domain-specific noise)

`cluster_id_seq = [9, 3]`, L = 2, support = 3438, n_distinct_repos = 11.
Texts:
- `When method post`
- `Then status 200`

**Label A:** no &mdash; while the recurrence is real and cross-repo, the slice
is too low-content to warrant a named step. It's the canonical Karate
&ldquo;send a POST and check 200&rdquo; pair; promoting it to a named higher-level
step would just rename a 2-line shorthand into another 2-line shorthand.

(Borderline call. Calibrate during pilot.)

### Example 4 (flagged-spec)

`cluster_id_seq = [52, 53]`, L = 2, support = 1770, max_within_file_recurrence = 1740.
Texts:
- `Given every active "memorydb" "cluster" has write durability enabled`
- `And every snapshotting "memorydb" "cluster" has a corresponding in-progress "memorydb" "snapshot"`

**Label:** `flagged-spec` &mdash; from the local-web-services memorydb spec
suite. Recurrence is a generator artefact. Excluded from headline
percentages.

### Example 5 (no / N-1 incoherent boundary)

`cluster_id_seq = [11, 13]`, L = 2, support = 3495.
Texts:
- `Then the result should be, in any order:`
- `And no side effects`

**Label A:** no &mdash; this is the *closing* of an OpenCypher TCK test pattern.
The slice cuts the assertion-block short; the &ldquo;result should
be&rdquo; line is a doc-string anchor and the &ldquo;no side effects&rdquo;
is a bullet-list anchor. Treating these two as a unit is a syntactic
accident from the way Gherkin parses doc strings.

---

## 6. Anti-rubric: things we are NOT trying to label

- **Whether the test is good.** Out of scope; we label extractability,
  not test quality.
- **Whether the underlying step-definition implementation is correct.**
  We do not have the step-definition source code in the corpus; we cannot
  judge implementation.
- **Whether the maintainer would actually accept the patch.** Real-world
  acceptance is a separate question (Phase 5/10 of the paper-2 plan
  addresses this). The rubric labels intrinsic extractability, not
  acceptance probability.

---

## 7. Open items for pilot review

- [ ] Should `unsure` count as 0.5 yes / 0.5 no for kappa, or as a
  third category? (Cukereuse paper 1 used three-category kappa.)
- [ ] Should `flagged-spec` be detected automatically before labelling
  (so labellers don't waste time) or only by labellers? Pilot first.
- [ ] L=2 minimum &mdash; is it too low? Pilot may suggest L=3 minimum
  for &ldquo;non-trivial&rdquo;.
- [ ] How to present `cluster_id_seq` to labellers &mdash; raw ids,
  canonical text only, or both? Lean: canonical text + scope signals,
  hide cluster ids unless asked.
