import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "data/manifests"
bm25 = json.loads((manifest / "bioasq_bm25_chunk_comparison.json").read_text(encoding="utf-8"))
dense = json.loads((manifest / "bioasq_medcpt_chunk_comparison.json").read_text(encoding="utf-8"))
both_improve = all(report["metrics"]["recall_at_10"]["ci95"][0] > 0 for report in (bm25, dense))
result = {"selection_rule": "C2 only if paired Recall@10 improves for both BM25 and MedCPT",
          "comparisons": {"bm25": bm25, "medcpt": dense}, "selected": "C2" if both_improve else "C0",
          "reason": "C2 did not improve both retrievers" if not both_improve else "C2 improved both retrievers"}
(manifest / "bioasq_chunk_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"selected": result["selected"], "reason": result["reason"]}, indent=2))
