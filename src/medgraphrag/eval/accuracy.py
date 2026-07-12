from medgraphrag.core.types import Question, Prediction


def accuracy(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.answer for q in questions}
    if len(preds) != len(questions):
        raise ValueError("prediction/question count mismatch")
    correct = sum(1 for p in preds if p.choice == gold.get(p.qid))
    return correct / len(questions)
