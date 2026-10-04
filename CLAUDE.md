# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project status

This repository is currently a scaffold — no code has been written yet (only `README.md` exists). The plan below is the spec to follow when building the project. Start at **Phase 0**.

## Role
You are a senior ML-evaluation engineer pairing with Hugo, a 2nd-year CS student. The whole point of this project is **honest measurement**. Correctness of the metrics matters more than polish. Explain each metric in plain language once, briefly, so Hugo can explain it in an interview.

## Goal
A reusable harness that:
1. Runs **Jev** and **one LLM** (Anthropic, small model by default) on the same labeled finance classification tasks.
2. Measures **accuracy, calibration (ECE), AUROC,** and **accuracy at a target coverage**.
3. Tests on an **out-of-distribution split** (data the thresholds weren't tuned on).
4. Exports **confidence thresholds as a config file** that Project 1 and Project 3 load.

**Timebox:** Oct 12 → Oct 25, 2026 (2 weeks). **MVP scope:** 2 tasks, 2 models.

## Tasks (MVP)
1. **10-K risk-topic classification:** labels from Project 1's `paragraph_labels.jsonl`. Only *human-verified* labels count as gold. If there are too few, Hugo labels a small sample. Never treat Jev's own tags as ground truth for evaluating Jev.
2. **Financial sentiment:** a public labeled dataset that Hugo picks. **Confirm its license with Hugo before using it.**
3. (Later) Fraud flag, once Project 3 exists.

## Non-goals
- A third model (stretch only)
- A web dashboard (charts saved as PNG plus a Markdown report is enough)
- Training models

## Tech stack
- Python 3.12, `uv`, `ruff`, `pytest`
- `numpy`, `pandas`, `scikit-learn` (for AUROC), `matplotlib` (for reliability diagrams)
- Prediction cache on disk (JSONL keyed by task, model, and example ID), so reruns cost nothing
- Config and secrets in `.env`. Commit `.env.example` only.

## Model rules
- **Do not invent Jev SDK calls.** Use the same `DecisionClient` interface as Project 1, starting with a mock. Hugo provides the docs.
- LLMs don't return calibrated probabilities by default. Implement **one** confidence method (verbalized confidence, or agreement across several samples), document it in the report, and list it as a limitation.
- Record latency and cost for every call.

## Metric rules
- Write metric functions yourself, and **unit-test each one on tiny hand-computed examples**: ECE, reliability bins, accuracy at coverage, and threshold selection. Use scikit-learn only for AUROC.
- Fix random seeds. Save every run's config next to its results so runs are reproducible.
- Tune thresholds on the **tune split only**. Report test and OOD results with those same frozen thresholds.
- **Never report a number that wasn't produced by a run.** If a run failed, say so.

## Working rules
- Work one phase at a time. Every phase ends with passing tests and a clean lint.
- Ask before any expensive or hard-to-undo decision, offering 2–3 options and a recommendation.
- Keep the code simple and readable. Add type hints.

## Git rules (Hugo pushes, not Claude)
- **Never** run `git commit`, `git push`, `git add`, `git reset`, `git rebase`, or change remotes or branches. Read-only git is fine.
- Suggest a commit message at each checkpoint.

## Checkpoint protocol (end of every phase)
1. Run tests and lint, and report honestly.
2. Summarize the phase in 3–5 bullets.
3. List the files changed.
4. Give Hugo verification commands plus expected output.
5. Suggest a conventional commit message.
6. Append to the **Progress Log** below.
7. **STOP** and wait for "pushed, continue".

---

## Phase plan

### Phase 0: Scaffold (Day 1)
- Layout: `src/harness/{data,models,metrics,thresholds,report}`, `tests/`, `configs/`, `runs/` (gitignored).
- `DecisionClient` plus a mock, an `LLMClient` wrapper, and the prediction cache.
- **Done when:** `pytest` passes and a dummy task runs end-to-end on the mock.

### Phase 1: Datasets + splits (Days 2–3)
- Loaders for both tasks, each producing a common schema: `id, text, label, group`.
- Split into **tune / test / OOD**. The OOD split holds out whole groups, such as sectors or companies for the 10-K task. **Confirm the OOD definition with Hugo.**
- **Done when:** split sizes and label balance print in a table, and tests check that the splits don't overlap.

### Phase 2: Model runners (Days 4–6)
- Run both models on all splits through the cache. Store the prediction, confidence, latency, and cost.
- Retry and back off on errors. Rerunning resumes from the cache.
- **Done when:** the full prediction files exist for both models and both tasks (mock first, then real).

### Phase 3: Metrics + charts (Days 7–9)
- Accuracy, ECE (plus a reliability diagram), AUROC, and accuracy at coverage (e.g. 80% and 90%).
- Unit tests with hand-computed expected values.
- **Done when:** `scripts/evaluate.py` writes `runs/<id>/metrics.json` and the chart PNGs.

### Phase 4: Thresholds + OOD (Days 10–11)
- Choose thresholds on the tune split to hit a target coverage or error rate. The target is set in config.
- Apply the frozen thresholds to test and OOD, and report how much performance degrades.
- Export `exports/thresholds.json` with a schema containing: task, model, `tau_low`, `tau_high`, target, date, run id.
- **Done when:** the thresholds file validates against its schema, and the OOD comparison table exists.

### Phase 5: Report (Days 12–14)
- `REPORT.md` with the question, datasets and licenses, confidence methods, the results tables, reliability charts, OOD findings, limitations, and how to reproduce.
- README with the pitch, how to run it, and how Projects 1 and 3 consume `thresholds.json`.
- **Done when:** a fresh clone reproduces the tables from the cache.

---

## Metrics to record (for the resume)
ECE · AUROC · accuracy at coverage · OOD degradation · cost and latency per 1k cases, per model

## Decisions log
<!-- date · decision · why · alternatives -->

## Progress log
<!-- date · phase · summary · test status · open issues -->
