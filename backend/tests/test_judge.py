import pytest
import json
from concurrent.futures import ThreadPoolExecutor

from app.judge import GatewayJudge, correctness_input, faithfulness_input, validate_judgement


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
    masked = correctness_input("q", "gold", "PrimeKG structured evidence supports this", evidence)
    assert "PrimeKG" not in masked["candidate_answer"]
    assert "structured evidence" not in masked["candidate_answer"]
    with pytest.raises(ValueError):
        validate_judgement("correctness_completeness", {"correctness": 9})
    with pytest.raises(ValueError):
        validate_judgement("correctness_completeness", {
            "claims": {}, "justification": "", "correctness": 1,
            "completeness": 1, "confidence": .5,
        })
    for invalid in (True, 1.5):
        with pytest.raises(ValueError):
            validate_judgement("correctness_completeness", {
                "claims": [], "justification": "bad scale", "correctness": invalid,
                "completeness": 1, "confidence": .5,
            })


def test_judge_deduplicates_concurrent_identical_cache_misses(tmp_path, monkeypatch):
    prompt = tmp_path / "configs/prompts/judge_correctness_completeness_v1.txt"
    prompt.parent.mkdir(parents=True); prompt.write_text("rubric", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    calls = []

    class Response:
        def raise_for_status(self): pass
        def json(self):
            parsed = {"claims": [], "justification": "ok", "correctness": 2,
                      "completeness": 2, "confidence": 1.0}
            return {"choices": [{"message": {"content": json.dumps(parsed)}}]}

    monkeypatch.setattr("app.judge.httpx.post", lambda *args, **kwargs: calls.append(1) or Response())
    judge = GatewayJudge(tmp_path, model="judge-test")
    payload = correctness_input("q", "gold", "candidate")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: judge.evaluate(payload), range(4)))
    assert len(calls) == 1
    assert all(result["correctness"] == 2 for result in results)
