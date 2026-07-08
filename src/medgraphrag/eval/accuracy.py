from medgraphrag.core.types import Question, Prediction


def accuracy(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.answer for q in questions}
    if len(preds) != len(questions):
        raise ValueError("prediction/question count mismatch")
    correct = 0
    for p in preds:
        if p.qid not in gold:
            raise ValueError(f"no gold for qid {p.qid}")
        if p.choice == gold[p.qid]:
            correct += 1
    return correct / len(questions)
