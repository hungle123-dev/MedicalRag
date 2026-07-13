from app.pipelines import evidence_budget


def test_evidence_budget_is_bounded_and_text_first_interleaved():
    texts = [{"id": f"t{i}", "snippet": "text " * 300} for i in range(8)]
    graphs = [{"id": f"g{i}", "snippet": "graph " * 200} for i in range(5)]
    selected, log = evidence_budget(texts, graphs)
    assert len(selected) <= 8
    assert log["graph_tokens_actual"] <= 540
    assert log["graph_tokens_actual"] + log["text_tokens_actual"] <= 1800
    assert selected[0]["id"].startswith("t")
