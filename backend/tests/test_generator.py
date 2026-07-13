import json

from app.generator import GatewayGenerator, MockGenerator, validate_citations


def test_mock_generator_only_cites_registered_evidence():
    evidence = [{"id": "PMID:1", "type": "text", "snippet": "Supported statement."}]
    answer = MockGenerator().generate("Question?", evidence).answer
    assert validate_citations(answer, evidence) == {"valid_ids": ["PMID:1"], "invented_ids": [], "valid": True}


def test_invented_citation_is_detected():
    result = validate_citations("Claim [PMID:999]", [{"id": "PMID:1"}])
    assert result["invented_ids"] == ["PMID:999"]
    assert not result["valid"]


def test_comma_separated_citations_are_validated_individually():
    evidence = [{"id": "PMID:1"}, {"id": "PMID:2"}]
    assert validate_citations("Claim [PMID:1, PMID:2]", evidence)["valid_ids"] == ["PMID:1", "PMID:2"]


def test_gateway_generator_uses_openai_compatible_endpoint(tmp_path, monkeypatch):
    prompts = tmp_path / "configs/prompts"; prompts.mkdir(parents=True)
    (prompts / "answer_v1.txt").write_text("{question}\n{evidence}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1/")
    monkeypatch.setenv("GATEWAY_GENERATOR_MODEL", "test-model")

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "Supported [PMID:1]"}}]}

    request = {}
    def post(url, **kwargs): request.update(url=url, **kwargs); return Response()
    monkeypatch.setattr("app.generator.httpx.post", post)
    result = GatewayGenerator(tmp_path).generate(
        "Question?", [{"id": "PMID:1", "type": "text", "snippet": "Supported"}]
    )
    assert result.answer == "Supported [PMID:1]"
    assert request["url"] == "https://gateway.example/v1/chat/completions"
    assert request["json"]["model"] == "test-model"
    assert json.loads(next((tmp_path / "artifacts/model_cache/gateway").glob("*.json")).read_text())["answer"] == result.answer
