"""Loader for the financial sentiment task: Financial PhraseBank,
`Sentences_AllAgree.txt` (2,264 sentences, 100% annotator agreement).

CC-BY-NC-SA-3.0, non-commercial (Malo & Sinha) -- fine for this project, but
don't redistribute commercially without contacting the authors.

Downloaded straight from the authors' own release (`takala/financial_phrasebank`
on the Hugging Face Hub) rather than the `datasets` library's loader for it,
which depends on a dataset-loading script format the library no longer
supports. Parsing it ourselves also avoids relying on a third-party remix
with its own, non-canonical train/test split.

No company/sector/date metadata exists in this dataset, so there's no real
group to hold out for an OOD split -- `group` is just each example's own id.
Use `split_random`, never `split_by_group`, for this task; the lack of a
true OOD split for sentiment is a documented limitation, not an oversight.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable

from harness.data.schema import Example

TASK = "financial_sentiment"
HF_REPO = "takala/financial_phrasebank"
ZIP_PATH_IN_REPO = "data/FinancialPhraseBank-v1.0.zip"
TXT_PATH_IN_ZIP = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"


def load() -> list[Example]:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=ZIP_PATH_IN_REPO)
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(TXT_PATH_IN_ZIP).decode("iso-8859-1")
    lines = (line for line in raw.splitlines() if line.strip())
    return _to_examples(lines)


def _to_examples(lines: Iterable[str]) -> list[Example]:
    examples = []
    for i, line in enumerate(lines):
        sentence, label = line.rsplit("@", 1)
        example_id = f"fpb-{i}"
        examples.append(Example(id=example_id, text=sentence, label=label, group=example_id))
    return examples
