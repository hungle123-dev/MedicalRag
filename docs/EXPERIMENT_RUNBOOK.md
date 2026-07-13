# MedicalRAG experiment runbook

Tài liệu này là runbook vận hành. Thiết kế nghiên cứu đầy đủ nằm trong
`KE_HOACH_THUC_NGHIEM_MEDICAL_RAG.html`.

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
uv run medrag experiment retrieval --method rrf --population selection4849
uv run medrag experiment retrieval --method rrf_rerank --population selection4849
```

Primary metric là BioASQ MAP@10; Recall@10/100, MRR, nDCG, latency và failure rate là secondary.
Chỉ thay đúng retriever hoặc indexed field trong từng contrast. MedCPT revision được pin trong
`artifacts/indexes/medcpt/metadata.json`.

## 4. Generation và oracle

```powershell
uv run medrag experiment generation --pipeline bm25_gpt41nano --population smoke40
uv run medrag experiment generation --pipeline bm25_gemini_flash_lite --population smoke40
uv run medrag experiment generation --pipeline bm25_deepseek --population smoke40
uv run medrag experiment generation --pipeline bm25_qwen --population smoke40
uv run medrag experiment oracle --population validation200
```

Sau smoke, các generator qua gate mới chạy `generation160`, rồi finalists chạy `validation200`.
Khi so generator, serialized evidence, prompt, parser, output budget và IDs phải giống hệt. Oracle:

- L0: closed-book;
- L1: predicted documents + predicted snippets;
- L2: gold documents + predicted snippets;
- L3: gold snippets.

Oracle chỉ định vị bottleneck, không phải pipeline deployable.

## 5. Statistics và held-out

```powershell
uv run medrag experiment compare --left artifacts/runs/LEFT/predictions.jsonl `
  --right artifacts/runs/RIGHT/predictions.jsonl --metric metrics.ap
uv run medrag experiment freeze-final
uv run medrag experiment freeze-final --verify
```

So sánh dùng paired normalized-group bootstrap CI, paired effect size và permutation p-value.
Final chỉ có đúng ba contrast preregistered và Holm family size 3. Chỉ chạy `heldout340` sau khi
five arms, pipeline configs, model inventory, judges và contrasts đã được freeze. Sau khi xem final
result không được sửa pipeline.

## 6. Product và observability

```powershell
uv run uvicorn apps.api.main:app --reload
cd apps/web
npm ci
npm run dev
```

Mỗi answer có trace ID, ranked PMIDs, packed evidence, prompt hash, model, token, latency, retry và
validated citations. Provider failure không bị drop khỏi experiment denominator. Model response
được cache bằng request hash; secret không nằm trong trace, report hay Git.
