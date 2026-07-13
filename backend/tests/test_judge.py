from app.judge import correctness_input, faithfulness_input


def test_judge_passes_are_information_separated():
    correctness = correctness_input("q", "gold", "candidate")
    faithfulness = faithfulness_input("candidate", [{"id": "PMID:1"}])
    assert "cited_evidence" not in correctness
    assert "reference_answer" not in faithfulness
    assert "pipeline_id" not in correctness and "pipeline_id" not in faithfulness
