from app.controls import (matched_extra_text, matched_graph_control_audit,
                          matched_random_paths, no_path_reranker, one_hop)


def test_graph_controls_preserve_requested_hops():
    paths = [{"id": "b", "hop_count": 1, "snippet": "one"},
             {"id": "a", "hop_count": 2, "snippet": "one two"},
             {"id": "c", "hop_count": 2, "snippet": "one two three"}]
    assert one_hop(paths) == [paths[0]]
    assert no_path_reranker(paths, 1)[0]["id"] == "a"
    sampled = matched_random_paths(paths, [{"id": "target", "hop_count": 2, "snippet": "one two"}], seed=1)
    assert sampled[0]["hop_count"] == 2
    extra = matched_extra_text([{"id": "t1", "snippet": "one two three"}],
                               [{"id": "g1", "snippet": "graph graph"}])
    assert extra[0]["snippet"] == "one two"


def test_extra_text_skips_short_ranked_candidate_without_reusing_items():
    candidates = [{"id": "short", "type": "text", "snippet": "one"},
                  {"id": "long", "type": "text", "snippet": "one two three four"},
                  {"id": "next", "type": "text", "snippet": "five six seven"}]
    targets = [{"id": "g1", "snippet": "a b c"}, {"id": "g2", "snippet": "d e"}]
    selected = matched_extra_text(candidates, targets)
    assert [item["id"] for item in selected] == ["long", "next"]
    assert [len(item["snippet"].split()) for item in selected] == [3, 2]


def test_random_path_stays_structural_and_post_budget_audit_detects_missing_slot():
    targets = [{"id": "g1", "type": "graph", "hop_count": 1, "snippet": "a —r→ b",
                "nodes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                "edges": [{"source_id": 1, "target_id": 2, "relation": "r"}]},
               {"id": "g2", "type": "graph", "hop_count": 1, "snippet": "c —r→ d",
                "nodes": [{"id": 3, "name": "c"}, {"id": 4, "name": "d"}],
                "edges": [{"source_id": 3, "target_id": 4, "relation": "r"}]}]
    candidates = [{"id": "r1", "type": "graph", "hop_count": 1, "snippet": "one —r→ two words",
                   "nodes": [{"id": 5, "name": "one"}, {"id": 6, "name": "two words"}],
                   "edges": [{"source_id": 5, "target_id": 6, "relation": "r"}]},
                  {"id": "r2", "type": "graph", "hop_count": 1, "snippet": "five —r→ six words",
                   "nodes": [{"id": 7, "name": "five"}, {"id": 8, "name": "six words"}],
                   "edges": [{"source_id": 7, "target_id": 8, "relation": "r"}]}]
    sampled = matched_random_paths(candidates, targets, seed=3)
    assert sampled[0]["snippet"] in {candidate["snippet"] for candidate in candidates}
    assert matched_graph_control_audit(targets, sampled)["complete"] is True
    incomplete = matched_graph_control_audit(targets, sampled[:1])
    assert incomplete["complete"] is False
    assert incomplete["requested_slots"] == 2
    assert incomplete["matched_slots"] == 1
