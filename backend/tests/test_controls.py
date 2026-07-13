from app.controls import matched_extra_text, matched_random_paths, no_path_reranker, one_hop


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
