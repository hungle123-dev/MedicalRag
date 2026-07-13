from app.controls import matched_extra_text, matched_random_paths, no_path_reranker, one_hop


def test_graph_controls_preserve_requested_hops():
    paths = [{"id": "b", "hop_count": 1}, {"id": "a", "hop_count": 2}, {"id": "c", "hop_count": 2}]
    assert one_hop(paths) == [paths[0]]
    assert no_path_reranker(paths, 1)[0]["id"] == "a"
    sampled = matched_random_paths(paths, [{"id": "target", "hop_count": 2}], seed=1)
    assert sampled[0]["hop_count"] == 2
    assert len(matched_extra_text([{"snippet": "one two"}, {"snippet": "three"}], 3)) == 2
