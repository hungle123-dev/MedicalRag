import json

from medrag_lab.data.loaders import load_inference_questions


def test_inference_view_never_contains_gold(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Does treatment X improve outcome Y?",
                "answer": "gold prose",
                "relevant_passage_ids": ["123"],
                "snippets": [{"text": "gold"}],
                "type": "yesno",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    question = load_inference_questions(path)[0]
    assert question.model_dump() == {
        "question_id": "q1",
        "question": "Does treatment X improve outcome Y?",
    }
    assert question.model_config["extra"] == "forbid"
