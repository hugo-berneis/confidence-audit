# confidence-audit: a finance calibration harness

**Status: pipeline validated on mocks. No real-model results yet.** Everything
below is real output from this repo, reproducible by anyone who clones it —
but it validates that the harness runs end-to-end correctly, not that Jev or
the LLM are well-calibrated. See [Pending real runs](#pending-real-runs) for
exactly what's missing and why.

## The question

Jev (TypeSafe's "System One" decision model) and an LLM both report a
confidence alongside every decision. Project 1 ([10k-analyst](../10k-analyst))
already uses Jev's confidence to gate retrieval and answer display. This
harness asks: **is that confidence trustworthy?** Specifically, on two finance
classification tasks:

1. How accurate is each model?
2. How well does its stated confidence track its actual correctness
   (calibration — ECE, and a reliability diagram)?
3. How well does confidence separate right answers from wrong ones
   (discrimination — AUROC)?
4. If you only answer the fraction of cases you're most confident about, how
   much more accurate do you get (accuracy at coverage)?
5. Do thresholds tuned on one slice of data still hold up on a slice the
   tuning never saw (out-of-distribution degradation)?

## Tasks and datasets

### 1. 10-K risk-topic classification

Paragraphs from Project 1's SEC 10-K filings (`risk_factors` and `mdna`
sections, 6 companies: AAPL, KO, MSFT, PFE, V, WMT), classified into one of
11 topics (`liquidity`, `credit`, `regulatory`, `cyber`, `macro`,
`competition`, `operations`, `legal`, `supply_chain`, `labor`, `other` — the
same set Jev is restricted to in Project 1, from its
`config/topics.yaml`).

**Gold labels:** Project 1's own tagger output
(`exports/paragraph_labels.jsonl`) could not be used — it's Jev's own tags
(every row has `topic_confidence == 0.6`, the real Jev client's constant
return on that run), and evaluating Jev against its own tags would be
circular. Instead, `scripts/prepare_10k_gold_sample.py` drew a 131-row
stratified sample (by ticker × section) from Project 1's *raw, untagged*
paragraphs into `data/gold/risk_topic_sample.jsonl`, for Hugo to hand-label
(instructions in `data/gold/README.md`). **This sample is not yet labeled**,
so this task has no results in this report.

### 2. Financial sentiment

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
`Sentences_AllAgree` subset: 2,264 sentences from financial news, each
labeled `positive` / `negative` / `neutral` by multiple annotators with 100%
agreement. Downloaded directly from the authors' release
(`takala/financial_phrasebank` on the Hugging Face Hub) rather than through
the `datasets` library, whose loader for this dataset depends on a
loading-script format the library no longer supports.

**License: CC-BY-NC-SA-3.0** (Malo & Sinha) — non-commercial, attribution,
share-alike. Fine for this project; don't redistribute commercially without
contacting the authors.

## Confidence methods

- **Jev:** native confidence from `DecisionClient.choice()` — the same
  interface Project 1 uses, mirrored (not reimplemented) in
  `src/harness/models/decision.py`.
- **LLM (Anthropic, `claude-haiku-4-5-20251001` by default):** LLMs don't
  return calibrated probabilities by default, so this harness implements
  **verbalized confidence** — the model is asked to return
  `{"label": ..., "confidence": <0-100>}` as strict JSON, and the confidence
  is rescaled to 0–1. See `LLMClient.classify()`.

**Limitation:** a verbalized confidence is not the same as a calibrated
probability. The model is self-reporting a number; it was never trained
against a proper scoring rule to make that number meaningful. ECE and AUROC
below measure whether it *happens* to track correctness, not whether the
model has any real insight into its own reliability.

## Results (mock run only — see caveat above)

Both the mock Jev client (`src/harness/models/decision.py`) and the mock LLM
client (`src/harness/models/llm_client.py`) return a **constant** confidence
regardless of input (0.6 and 0.7 respectively) — deliberately, so tests are
deterministic. That has a direct consequence for every number below: a
constant confidence carries no information, so ECE, AUROC, and the
threshold bands are measuring that fact, not a real model property.

**Financial sentiment**, from `runs/20261004T200426Z/metrics.json`:

| model    | split | n    | accuracy | ECE   | AUROC | acc @ 80% coverage | acc @ 90% coverage |
|----------|-------|------|----------|-------|-------|---------------------|---------------------|
| jev-mock | tune  | 1585 | 0.334    | 0.266 | 0.500 | 0.344               | 0.338               |
| jev-mock | test  | 679  | 0.325    | 0.275 | 0.500 | 0.317               | 0.334               |
| llm-mock | tune  | 1585 | 0.338    | 0.362 | 0.500 | 0.333               | 0.333               |
| llm-mock | test  | 679  | 0.333    | 0.367 | 0.500 | 0.341               | 0.342               |

Accuracy sits at ~1/3 on a 3-class task — expected, since both mocks pick a
label via a hash of the input text, uncorrelated with the true label.
AUROC of exactly 0.500 is the mathematical signature of a constant
confidence score: with no variation to rank examples by, ROC-AUC reduces to
chance. Reliability-diagram PNGs for every (model, split) pair are in
`runs/20261004T200426Z/charts/`.

**Accuracy-at-coverage vs. accuracy** is a useful sanity check even on mock
data: accuracy among the "most confident" 80%/90% barely differs from plain
accuracy, because — again — there's no real confidence signal to rank by.
On real data, a well-calibrated model should show accuracy *rising* as
coverage shrinks.

**Thresholds** (`exports/thresholds.json`, validated against
`configs/thresholds.schema.json`), tuned on tune only (target coverage 0.8,
target error rate 0.1), applied unchanged to test:

| model    | tau_low | tau_high |
|----------|---------|----------|
| jev-mock | 0.6     | 0.6      |
| llm-mock | 0.7     | 0.7      |

`tau_low == tau_high` for both models — again the constant-confidence
artifact: with only one observed confidence value, there's nothing to carve
into separate "reject" / "trust" bands. `exports/thresholds_report.json` has
the full tune-vs-test comparison; with these degenerate thresholds, coverage
is 1.0 and accuracy is unchanged between tune and test for both models (no
OOD comparison exists yet — the 10-K task is the one with an OOD split, and
it has no gold labels).

## OOD findings

**None yet.** The confirmed OOD design (hold out tickers PFE and V entirely
from `risk_topic`'s tune/test) only applies to the 10-K task, which has no
gold labels. The sentiment task has no natural grouping to define an OOD
split on (Financial PhraseBank carries no company/sector/date metadata) —
documented as a limitation, not an oversight, when Hugo confirmed this in
Phase 1.

## Limitations

- **No real model results.** Every number above comes from deterministic
  mocks with constant confidence. Real Jev and real Anthropic calls need:
  (1) `data/gold/risk_topic_sample.jsonl` hand-labeled (131 rows, see
  `data/gold/README.md`), and (2) a `.env` with real API keys (see
  `.env.example`). The real `JevDecisionClient` also isn't wired up yet —
  Phase 0 scoped only the mock; wiring the real client mirrors Project 1's
  TypeSafe SDK pattern in `report_qa/decision_jev.py`.
- **Verbalized confidence, not calibrated probability** (see Confidence
  methods above) — a documented limitation of the one confidence method
  implemented for the LLM, not a bug.
- **No true OOD split for the sentiment task** — Financial PhraseBank has no
  grouping metadata to hold out.
- **131-row gold sample for the 10-K task.** Small by construction (MVP
  timebox); per-topic counts will be uneven since some topics are rarer in
  the underlying filings.
- **A third model, a web dashboard, and model training are explicit
  non-goals** of this MVP.

## How to reproduce

The prediction cache (`runs/cache/predictions.jsonl`) is **not** committed —
reproducing means rerunning the (free, deterministic, seeded) mock clients,
not pulling a cached artifact. This will need revisiting once real API
calls are in the mix, since rerunning those on every clone would cost money.

```sh
uv sync
uv run pytest -q                      # 81 passed
uv run scripts/run_models.py          # mock Jev + mock LLM on financial_sentiment
uv run scripts/evaluate.py            # writes runs/<run_id>/metrics.json + charts
uv run scripts/export_thresholds.py   # writes exports/thresholds.json + report
```

Same seed (`seed: int = 0` in `src/harness/config.py`), same deterministic
mocks → identical tables every time, under a new `run_id`.
