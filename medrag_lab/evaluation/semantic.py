from __future__ import annotations

from collections.abc import Sequence


def rouge2(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    from rouge_score.rouge_scorer import RougeScorer

    scorer = RougeScorer(["rouge2"], use_stemmer=True)
    return [
        scorer.score(reference, prediction)["rouge2"].fmeasure
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def bertscore(
    predictions: Sequence[str], references: Sequence[str], device: str | None = None
) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return []
    import torch
    from bert_score import score

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, f1 = score(
        list(predictions),
        list(references),
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        device=selected_device,
        batch_size=4 if selected_device == "cuda" else 2,
        verbose=False,
    )
    return [float(value) for value in f1.cpu().tolist()]
