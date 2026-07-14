# MedicalRAG experiment runbook

Tài liệu này là runbook vận hành. Thiết kế nghiên cứu đầy đủ nằm trong
`KE_HOACH_THUC_NGHIEM_MEDICAL_RAG.html`.

Implementation status: E11 is the only confirmatory answer-level stage. E05/E08/E09 stopped at
`generation160`; their planned `validation200` confirmation was not completed. ROUGE-SU4 and
snippet F1 are internal BioASQ-compatible implementations, not official leaderboard scorers.
The LLM panel failed its reliability gate and is diagnostic only.

## 1. Boundary dữ liệu

- Một bundle duy nhất: BioASQ-12B-RAG gồm `corpus.jsonl` (49.513 abstract), `dev.jsonl`
  (5.049 QA) và `eval.jsonl` (340 QA held-out).
- Corpus là positive-only, gold-conditioned candidate pool. Kết quả retrieval chỉ có ý nghĩa
  trong closed corpus này, không đại diện cho toàn PubMed.
- `dev` cung cấp câu hỏi, ideal answer, relevant PMID và gold snippet. Local bundle không có
  official exact-answer labels; vì vậy exact-answer metric bị khóa tắt.
- `selection4849`, `validation200`, `query800`, `generation160`, `judge160`, `smoke40` và
  `heldout340` được freeze theo normalized-question group. Không được đổi ID sau khi chạy arm.

## 2. Gate bắt buộc

```powershell
uv sync --extra dev --extra dense --extra semantic
uv run medrag data audit
uv run medrag data verify
uv run medrag experiment validate
uv run pytest
uv run ruff check .
```

Nếu hash raw/split sai, dừng. Smoke chỉ kiểm tra feasibility và failure; không dùng smoke quality
để chọn champion.

## 3. Retrieval tournament

```powershell
uv run medrag experiment bm25 --recipe title --population selection4849
uv run medrag experiment bm25 --recipe abstract --population selection4849
uv run medrag experiment bm25 --recipe title_abstract --population selection4849
uv run medrag experiment bm25 --recipe boosted_title_abstract_mesh --population selection4849

uv run medrag index build-medcpt --batch-size 16
uv run medrag experiment retrieval --method medcpt --population selection4849
uv run medrag experiment retrieval --method rrf --population selection4849 `
  --bm25-recipe boosted_title_abstract_mesh
uv run medrag experiment retrieval --method rrf_rerank --population selection4849 `
  --bm25-recipe boosted_title_abstract_mesh
```

Primary metric là BioASQ MAP@10; Recall@10/100, MRR, nDCG, latency và failure rate là secondary.
Chỉ thay đúng retriever hoặc indexed field trong từng contrast. MedCPT revision được pin trong
`artifacts/indexes/medcpt/metadata.json`.

## 4. Query và evidence tournament

Chỉ chạy E03 sau khi E02 đã có paired gate. `--retriever` phải đúng champion E02:

```powershell
uv run medrag experiment query --strategy original --population query800 --retriever rrf_rerank
uv run medrag experiment query --strategy mesh --population query800 --retriever rrf_rerank
uv run medrag experiment query --strategy hyde --population query800 --retriever rrf_rerank --workers 4
```

E04 luôn nhận **cùng một prediction file** để bốn arm thấy cùng ranked PMIDs:

```powershell
$retrieval = "artifacts/runs/E02-CHAMPION/predictions.jsonl"
uv run medrag experiment evidence --arm full_document_fields `
  --retrieval-predictions $retrieval --population selection4849
uv run medrag experiment evidence --arm fixed256_bm25 `
  --retrieval-predictions $retrieval --population selection4849
uv run medrag experiment evidence --arm sentence3_bm25 `
  --retrieval-predictions $retrieval --population selection4849
uv run medrag experiment evidence --arm sentence3_cross_encoder `
  --retrieval-predictions $retrieval --population selection4849
```

Snippet F1 nội bộ dùng character offsets theo định nghĩa BioASQ nhưng chưa cross-check official
scorer. Gold snippet chỉ được mở sau khi arm đã chọn evidence.

## 5. Context, generation, prompt và oracle

E05–E07 tạo artifact context gold-free trước. Thay đúng **một** flag mỗi contrast:

```powershell
uv run medrag experiment prepare-contexts --family E05 --arm budget600 `
  --pipeline rrf_rerank_rag --population generation160 --context-budget 600
uv run medrag experiment prepare-contexts --family E05 --arm budget1200 `
  --pipeline rrf_rerank_rag --population generation160 --context-budget 1200
uv run medrag experiment prepare-contexts --family E05 --arm budget2400 `
  --pipeline rrf_rerank_rag --population generation160 --context-budget 2400
uv run medrag experiment prepare-contexts --family E06 --arm strongest_middle `
  --pipeline rrf_rerank_rag --population generation160 --context-order strongest_middle
uv run medrag experiment prepare-contexts --family E07 --arm one_per_pmid `
  --pipeline rrf_rerank_rag --population generation160 --diversity one_per_pmid
```

Generator E08 dùng lại đúng một context file; `evidence_hash` phải bằng nhau trước paired comparison:

```powershell
$context = "artifacts/runs/E05-CHAMPION/contexts.jsonl"
uv run medrag experiment generate-contexts --family E08 --arm gpt41nano `
  --contexts $context --population generation160 --model gpt-4.1-nano --workers 4
uv run medrag experiment generate-contexts --family E08 --arm gemini_flash_lite `
  --contexts $context --population generation160 --model gemini-2.5-flash-lite --workers 4
uv run medrag experiment generate-contexts --family E08 --arm deepseek_v32 `
  --contexts $context --population generation160 --model deepseek-v3.2 --workers 4
uv run medrag experiment generate-contexts --family E08 --arm qwen35 `
  --contexts $context --population generation160 --model qwen3.5-122b-a10b --workers 4

uv run medrag experiment generate-contexts --family E09 --arm generic `
  --contexts $context --population generation160 --model MODEL_CHAMPION `
  --prompt-style generic_structured
uv run medrag experiment generate-contexts --family E09 --arm citation `
  --contexts $context --population generation160 --model MODEL_CHAMPION `
  --prompt-style citation_constraint
uv run medrag experiment generate-contexts --family E09 --arm predicted_type `
  --contexts $context --population generation160 --model MODEL_CHAMPION `
  --prompt-style predicted_type_schema
uv run medrag experiment generate-contexts --family E09 --arm gold_type_oracle `
  --contexts $context --population generation160 --model MODEL_CHAMPION `
  --prompt-style gold_type_oracle

uv run medrag experiment oracle --population validation200 --pipeline best_rag
```

Protocol dự kiến finalists chạy `validation200`, nhưng E05/E08/E09 thực tế chưa hoàn thành bước đó;
không được gọi winner từng module là independently confirmed.
Khi so generator, serialized evidence, prompt, parser, output budget và IDs phải giống hệt. Oracle:

- L0: closed-book;
- L1: predicted documents + predicted snippets;
- L2: gold documents + predicted snippets;
- L3: gold snippets.

Oracle chỉ định vị bottleneck, không phải pipeline deployable.

## 6. Statistics, panel, interaction và held-out

```powershell
uv run medrag experiment compare --left artifacts/runs/LEFT/predictions.jsonl `
  --right artifacts/runs/RIGHT/predictions.jsonl --metric metrics.ap
uv run medrag experiment compare --left artifacts/runs/MODEL_A/scored.jsonl `
  --right artifacts/runs/MODEL_B/scored.jsonl --metric rouge_su4.f1 `
  --require-equal-evidence
uv run medrag experiment interaction --id retriever_x_query `
  --a0b0 A0B0.jsonl --a0b1 A0B1.jsonl --a1b0 A1B0.jsonl --a1b1 A1B1.jsonl `
  --metric metrics.ap
uv run medrag experiment panel-direct --generation SCORED.jsonl --contexts CONTEXTS.jsonl `
  --population validation200 --workers 2
uv run medrag experiment freeze-final
uv run medrag experiment freeze-final --verify
uv run medrag experiment final-holm --comparison C1.json --comparison C2.json --comparison C3.json
```

So sánh dùng paired normalized-group bootstrap CI, paired effect size và permutation p-value.
Final chỉ có đúng ba contrast preregistered và Holm family size 3. Chỉ chạy `heldout340` sau khi
five arms, pipeline configs, model inventory, judges và contrasts đã được freeze. Sau khi xem final
result không được tái chấm hoặc đổi winner. Submission hardening có thể sửa runtime/report, nhưng
source guard sẽ khóa mọi held-out rerun; `freeze-final --verify` phải báo
`heldout_rerun_allowed=false` sau các thay đổi đó.

Panel ba model là proxy tự động, phải ghi đúng tên; không được gọi là human/physician review.
Observed disagreement/failure khiến reliability gate fail, nên không dùng panel để claim medical
correctness hoặc faithfulness.

## 7. Product và observability

```powershell
uv run uvicorn apps.api.main:app --reload
cd apps/web
npm ci
npm run dev
cd ../..
docker compose up --build
```

Best-RAG runtime rerank 100 documents nhưng chỉ đưa top 10 vào cross-encoder evidence selection,
khớp E04/E11 sealed recipe. Mỗi answer có trace ID, ranked PMIDs, packed evidence, prompt hash,
model, token, latency, retry và
validated citations. Provider failure không bị drop khỏi experiment denominator. Model response
được cache bằng request hash; secret không nằm trong trace, report hay Git.
