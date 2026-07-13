import pytest

from app.judge import correctness_input, faithfulness_input, validate_judgement


def test_judge_passes_are_information_separated():
    correctness = correctness_input("q", "gold", "candidate")
    faithfulness = faithfulness_input("candidate", [{"id": "PMID:1"}])
    assert "cited_evidence" not in correctness
    assert "reference_answer" not in faithfulness
    assert "pipeline_id" not in correctness and "pipeline_id" not in faithfulness


def test_faithfulness_input_removes_retrieval_scores():
    payload = faithfulness_input("answer", [{"id": "PMID:1", "snippet": "evidence", "score": 0.9}])
    assert payload["cited_evidence"] == [{"id": "E1", "snippet": "evidence"}]


def test_judge_inputs_mask_modality_and_validate_schema():
    evidence = [{"id": "primekg:path:secret", "type": "graph", "snippet": "A relates to B"}]
    payload = correctness_input("q", "gold", "claim [primekg:path:secret]", evidence)
    assert payload["candidate_answer"] == "claim [E1]"
    with pytest.raises(ValueError):
        validate_judgement("correctness_completeness", {"correctness": 9})
    with pytest.raises(ValueError):
        validate_judgement("correctness_completeness", {
            "claims": {}, "justification": "", "correctness": 1,
            "completeness": 1, "confidence": .5,
        })
