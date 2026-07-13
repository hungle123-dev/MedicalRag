from app.generator import MockGenerator, validate_citations


def test_mock_generator_only_cites_registered_evidence():
    evidence = [{"id": "PMID:1", "type": "text", "snippet": "Supported statement."}]
    answer = MockGenerator().generate("Question?", evidence).answer
    assert validate_citations(answer, evidence) == {"valid_ids": ["PMID:1"], "invented_ids": [], "valid": True}


def test_invented_citation_is_detected():
    result = validate_citations("Claim [PMID:999]", [{"id": "PMID:1"}])
    assert result["invented_ids"] == ["PMID:999"]
    assert not result["valid"]
