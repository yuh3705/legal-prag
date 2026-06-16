# Vietnamese Legal Parametric RAG Chatbot

Project demo **Parametric RAG kiểu PRAG** cho chatbot pháp luật Việt Nam.

Ý tưởng chính:

- RAG thường: retrieve văn bản rồi nhét text vào prompt.
- PRAG: retrieve văn bản rồi load **LoRA adapter** đã train cho văn bản đó vào LLM.
- Project này dùng `meta-llama/Llama-3.2-1B-Instruct`.
- Mỗi chunk pháp luật có thể được train thành một LoRA adapter riêng.
- Khi người dùng hỏi, hệ thống retrieve chunk liên quan, chọn LoRA adapter tương ứng, gắn adapter vào Llama rồi generate câu trả lời.

Vì mục tiêu là đồ án chạy được, mặc định chỉ train một số adapter nhỏ bằng `--max-adapters`. Không nên train toàn bộ dataset ngay từ đầu.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
huggingface-cli login
```

`huggingface-cli login` cần thiết nếu model Llama yêu cầu quyền truy cập trên Hugging Face.

## 1. Tải Dataset Và Lưu JSON

Dataset: `th1nhng0/vietnamese-legal-documents`

Chỉ giữ văn bản có:

```text
tinh_trang_hieu_luc == "Còn hiệu lực"
```

Chạy:

```powershell
python -m legal_prag.download_dataset --limit 1000
```

Output:

```text
data/raw/legal_docs_active.json
```

## 2. Chunk Văn Bản

```powershell
python -m legal_prag.prepare_corpus --chunk-size 900 --chunk-overlap 120
```

Output:

```text
data/processed/chunks.json
```

## 3. Build BM25 Retriever

```powershell
python -m legal_prag.build_store
```

Output:

```text
artifacts/retriever.joblib
artifacts/parametric_store.joblib
```

`retriever.joblib` là BM25 retriever. `parametric_store.joblib` chỉ dùng làm memory phụ và fallback. Phần parametric chính là LoRA ở bước tiếp theo.

## 4. Train LoRA Adapters

Train thử 3 adapter đầu tiên:

```powershell
python -m legal_prag.train_lora_adapters --max-adapters 3 --epochs 1
```

Output:

```text
artifacts/lora_adapters/
artifacts/lora_adapters/manifest.json
```

Mỗi adapter ứng với một `chunk_id`. Đây là bước **parametric encoding**: nội dung chunk được nén vào tham số LoRA.

Nếu máy yếu, có thể giảm:

```powershell
python -m legal_prag.train_lora_adapters --max-adapters 1 --epochs 1 --max-length 512
```

## 5. Chạy Chatbot PRAG

```powershell
python -m legal_prag.chatbot
```

Mặc định chatbot dùng:

```text
Retrieve -> chọn LoRA adapter -> load adapter vào Llama -> Generate
```

Nếu muốn chạy base Llama không dùng LoRA:

```powershell
python -m legal_prag.chatbot --no-lora
```

Nếu chỉ muốn test retrieval nhanh, không tải Llama:

```powershell
python -m legal_prag.chatbot --no-llm
```

## 6. Chạy Web UI

Chạy web demo nhanh, không tải Llama:

```powershell
python -m legal_prag.web_app --no-llm --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Chạy web với PRAG LoRA:

```powershell
python -m legal_prag.web_app --port 8000
```

Chạy web với base Llama nhưng không load LoRA:

```powershell
python -m legal_prag.web_app --no-lora --port 8000
```

## Pipeline Đồ Án

```text
Offline
Dataset Hugging Face
-> filter văn bản còn hiệu lực
-> save JSON
-> chunk văn bản
-> build BM25 retriever
-> self-augmentation đơn giản từ mỗi chunk
-> train LoRA adapter riêng cho từng chunk
-> save manifest: chunk_id -> adapter path

Online
User query
-> retrieve top-k chunks
-> tìm chunk có LoRA adapter
-> load adapter tương ứng vào Llama-3.2-1B-Instruct
-> generate answer
-> trả lời kèm nguồn retrieve
```

## Cấu Trúc

```text
legal_prag/
  download_dataset.py
  prepare_corpus.py
  build_store.py
  train_lora_adapters.py
  lora_generator.py
  chatbot.py
  web_app.py
web/
  index.html
  styles.css
  app.js
data/
  raw/
  processed/
artifacts/
  lora_adapters/
```

## Ghi Chú Thành Thật

Repo PRAG gốc có self-augmentation bằng LLM và merge nhiều LoRA adapter phức tạp hơn. Project này là bản đồ án tối giản:

- Có train LoRA thật.
- Có lưu adapter theo chunk.
- Có retrieve rồi load adapter tương ứng khi inference.
- Chưa tối ưu chất lượng, tốc độ, hoặc multi-adapter merging.
