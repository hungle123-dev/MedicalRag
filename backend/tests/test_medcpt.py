from app.medcpt import reciprocal_rank_fusion


def test_rrf_rewards_documents_found_by_both_retrievers():
    lexical = [{"id": "a", "retriever": "bm25"}, {"id": "b", "retriever": "bm25"}]
    dense = [{"id": "b", "retriever": "medcpt"}, {"id": "c", "retriever": "medcpt"}]
    fused = reciprocal_rank_fusion(lexical, dense)
    assert fused[0]["id"] == "b"
    assert fused[0]["retrievers"] == ["bm25", "medcpt"]
