from app.pipelines import build_e5_arms, evidence_budget


def test_evidence_budget_is_bounded_and_text_first_interleaved():
    texts = [{"id": f"t{i}", "snippet": "text " * 300} for i in range(8)]
    graphs = [{"id": f"g{i}", "snippet": "graph " * 200} for i in range(5)]
    selected, log = evidence_budget(texts, graphs)
    assert len(selected) <= 8
    assert log["graph_words_actual"] <= 540
    assert log["graph_words_actual"] + log["text_words_actual"] <= 1800
    assert selected[0]["id"].startswith("t")


def test_e5_arms_match_actual_word_budget_and_control_slots(monkeypatch):
    texts = [{"id": f"t{i}", "type": "text", "snippet": "text " * 10} for i in range(13)]
    target = [{"id": "g1", "type": "graph", "hop_count": 1, "snippet": "a —r→ b",
               "nodes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
               "edges": [{"source_id": 1, "target_id": 2, "relation": "r"}]},
              {"id": "g2", "type": "graph", "hop_count": 2,
               "snippet": "a —r→ b · b —s→ c",
               "nodes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"},
                         {"id": 3, "name": "c"}],
               "edges": [{"source_id": 1, "target_id": 2, "relation": "r"},
                         {"source_id": 2, "target_id": 3, "relation": "s"}]}]
    candidates = target + [
        {"id": "r1", "type": "graph", "hop_count": 1, "snippet": "x —r→ y",
         "nodes": [{"id": 4, "name": "x"}, {"id": 5, "name": "y"}],
         "edges": [{"source_id": 4, "target_id": 5, "relation": "r"}]},
        {"id": "r2", "type": "graph", "hop_count": 2,
         "snippet": "x —r→ y · y —s→ z",
         "nodes": [{"id": 4, "name": "x"}, {"id": 5, "name": "y"},
                   {"id": 6, "name": "z"}],
         "edges": [{"source_id": 4, "target_id": 5, "relation": "r"},
                   {"source_id": 5, "target_id": 6, "relation": "s"}]},
    ]
    monkeypatch.setattr("app.pipelines.text_evidence", lambda *args, **kwargs: texts)
    monkeypatch.setattr("app.pipelines.graph_evidence",
                        lambda *args, **kwargs: (candidates if kwargs.get("limit") == 100 else target, []))
    monkeypatch.setattr("app.pipelines.background_control_evidence",
                        lambda target_graphs, seeds, seed: candidates[2:])
    arms = build_e5_arms("question", seed=1)
    totals = {arm: sum(len(item["snippet"].split()) for item in value["evidence"])
              for arm, value in arms.items()}
    assert len(set(totals.values())) == 1
    assert len(arms["G2"]["evidence"]) == len(arms["X1"]["evidence"]) == len(arms["X2"]["evidence"])
    assert not {"g1", "g2"} & {item["id"] for item in arms["X2"]["evidence"]}
    assert arms["X1"]["control_complete"] is True
    assert arms["X2"]["control_complete"] is True
