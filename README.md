# Medical Graph RAG

Đồ án: so sánh các chiến lược Retrieval-Augmented Generation (RAG) — có và
không dùng knowledge graph — cho hỏi-đáp trắc nghiệm y khoa (MCQA).

## Bài toán

**Input:** câu hỏi trắc nghiệm y khoa (question + 4 lựa chọn A/B/C/D).
**Output:** 1 chữ cái lựa chọn (option key), kèm evidence đã dùng (nếu có
retrieval).

```
Question("Which agent is preferred for streptococcal pharyngitis?",
         options={"A": "penicillin", "B": "insulin", ...})
  -> Prediction(choice="A", evidence=[...])
```

## Dữ liệu (thật, không crawl)

| Nguồn | Nội dung | Kích thước |
|---|---|---|
| [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) | 5 subtask MCQA y khoa (MedQA, MedMCQA, PubMedQA, BioASQ, MMLU-Med) | 7,663 câu |
| [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) | 18 textbook y khoa, đã chunk sẵn | 125,847 chunk / 202MB |

Tải bằng `python scripts/download_data.py` (idempotent). Xem `data/eda_report.json`
cho thống kê (độ dài câu hỏi theo subtask, phân phối chunk...).

## Kiến trúc

Pipeline module hóa, mỗi thành phần đăng ký qua `core/registry.py` để thêm
biến thể mới không cần sửa code cũ:

```
src/medgraphrag/
├── core/        types (Question/Prediction/RetrievedItem), protocols, registry
├── data/        loader cho MIRAGE + corpus textbook
├── retrieval/   BM25 (bm25s, ~150ms/query trên 125k chunk)
├── llm/         client LLM qua endpoint OpenAI-compatible (chạy cả GPT & Gemini)
├── pipeline/    Arm (retriever+LLM) và runner
└── eval/        accuracy (answer) + retrieval_recall/mrr (tách biệt)
```

**Arm hiện có:**
- `E0` — LLM trả lời trực tiếp, không retrieval (đo kiến thức sẵn có của model).
- `E1` — BM25 retrieval trên corpus textbook trước khi hỏi LLM (RAG).

Xem [`docs/THIET_KE_Pipeline_MedicalGraphRAG.html`](docs/THIET_KE_Pipeline_MedicalGraphRAG.html)
cho thiết kế đầy đủ (10 giai đoạn pipeline, các arm graph/PPR dự kiến, ma trận
thực nghiệm, kế hoạch 5 tuần).

## Cách chạy

```bash
python -m pip install -e ".[dev]"
python scripts/download_data.py        # tải MIRAGE + corpus (1 lần)
python scripts/eda.py                  # thống kê dữ liệu
python scripts/build_index.py          # build BM25 index (cache pickle)
python scripts/make_subset.py          # lấy mẫu 900 câu stratified cho thực nghiệm
```

Chạy thực nghiệm thật (cần API key qua biến môi trường `OPENAI_API_KEY` +
`OPENAI_BASE_URL`, xem `.env.example`):

```bash
python scripts/run_experiment.py --limit 20     # smoke test
python scripts/run_experiment.py                # full 900 câu x 2 model x {E0,E1}
python scripts/analyze.py                        # bảng accuracy, McNemar, case study
```

Test (không cần key, chạy trên data thật đã tải):

```bash
pytest -q
```

## Trạng thái hiện tại

- Data thật đã tải + verify (7,663 câu, 125,847 chunk).
- Pipeline E0/E1 chạy được end-to-end, 30 test xanh.
- **Đang chạy thực nghiệm đầy đủ**: 900 câu × 2 model (gpt-4.1-nano,
  gemini-2.5-flash-lite) × {E0, E1} = 3,600 lời gọi LLM. Kết quả từng phần ở
  `results/*.jsonl` (đọc trực tiếp — có thể chưa đủ 900 dòng nếu đang chạy).
- Kết quả sơ bộ đáng chú ý (gpt-4.1-nano, đã xong 900×2): **E1 (BM25) thấp hơn
  E0** ở mọi subtask — retrieval hiện tại tạo nhiễu hơn tín hiệu (chỉ ~20% câu
  có đáp án đúng xuất hiện trong top-5 chunk). Đây là phát hiện thật, không
  phải lỗi — phân tích chi tiết trong `results/analysis.json` và báo cáo.
- Chưa làm (kế hoạch 5 tuần trong design doc): graph/PPR retrieval,
  entity-linking UMLS, fusion, rerank, fine-tune.

## Ghi chú cho người đọc code

- `tests/test_real_data_loaders.py`, `test_bm25_retriever.py` chạy trên
  **data thật** đã tải (tự skip nếu chưa tải).
- `llm/mock.py` (`MockLLM`) chỉ dùng để test wiring pipeline, KHÔNG dùng để
  báo cáo kết quả — mọi số accuracy chính thức đến từ `llm/openai_client.py`
  gọi LLM thật.
- Không commit `.env`, `data/raw/`, `data/bm25_index.pkl` (nặng/chứa key) —
  xem `.gitignore`. `scripts/download_data.py` + `build_index.py` tái tạo lại.
