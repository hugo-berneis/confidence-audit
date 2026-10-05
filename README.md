# confidence-audit

A calibration harness for Jev and an LLM: runs both on the same labeled
finance classification tasks and measures accuracy, calibration (ECE),
discrimination (AUROC), and accuracy at a target coverage — then exports
frozen confidence thresholds for [10k-analyst](../10k-analyst) (Project 1)
and Project 3 to load. See [REPORT.md](REPORT.md) for the full writeup
(current status: pipeline validated on mocks, real-model results still
pending).

## Author / Contact

- Hugo Berneis — hugo@berneis.com
- GitHub: HBerneis

## Instructions to Build

- Requires [uv](https://docs.astral.sh/uv/) installed first; `uv` installs the pinned Python 3.12 itself.
- Requires an [Anthropic API key](https://console.anthropic.com/) for real LLM runs (optional — the mock LLM client needs no key).

```bash
git clone https://github.com/<your-username>/confidence-audit.git
cd confidence-audit
cp .env.example .env       # add ANTHROPIC_API_KEY / JEV_API_KEY for real runs
uv sync                    # installs dependencies (Python 3.12, via uv)
```

## Instructions to Run

- Draw the 10-K gold sample (once), then hand-label it — see `data/gold/README.md`:

```bash
uv run scripts/prepare_10k_gold_sample.py
```

- Run both models (Jev + the LLM) on both tasks through the prediction cache:

```bash
uv run scripts/run_models.py
```

- Compute accuracy/ECE/AUROC/accuracy-at-coverage and write reliability charts:

```bash
uv run scripts/evaluate.py
```

- Freeze thresholds on tune, apply to test/OOD, and export `thresholds.json`:

```bash
uv run scripts/export_thresholds.py
```

All three scripts default to the mock Jev and mock LLM clients — free,
deterministic, and safe to run repeatedly (the prediction cache means
reruns cost nothing). Real calls need the API keys from `.env`.

## Exported Thresholds (for Project 1 and Project 3)

`scripts/export_thresholds.py` writes `exports/thresholds.json`, validated
against `configs/thresholds.schema.json`. It's a JSON array, one entry per
(task, model):

```json
[
  {
    "task": "financial_sentiment",
    "model": "llm-mock",
    "tau_low": 0.7,
    "tau_high": 0.7,
    "target": { "coverage": 0.8, "error_rate": 0.1 },
    "date": "2026-10-04",
    "run_id": "20261004T200428Z"
  }
]
```

- `tau_high` — above this confidence, trust the prediction outright.
- `tau_low` — below this confidence, reject/abstain; the prediction isn't
  reliable enough to show.
- Between the two — still shown, but flagged low-confidence.

This mirrors Project 1's existing `config/thresholds.yaml` pattern
(`relevance_drop_confidence` rejects below a threshold,
`groundedness_threshold` accepts above one) — never silently drop
something Jev is uncertain about. Both thresholds are tuned on the **tune
split only** and frozen before touching test/OOD; a consumer should treat
them as fixed operating points, not something to re-derive from its own
data.

`exports/` isn't committed in either repo (same convention as Project 1's
own `exports/`) — regenerate it by running the scripts above.

## Instructions to Run Test Suite(s)

```bash
uv run pytest
uv run ruff check .
```
