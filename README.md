# Vietnamese Legal Parametric RAG Chatbot

Project demo **Parametric RAG kiểu PRAG** cho chatbot pháp luật Việt Nam.

Ý tưởng chính:

- RAG thường: retrieve văn bản rồi đưa text vào prompt.
- PRAG: retrieve văn bản rồi load **LoRA adapter** đã train cho văn bản đó vào LLM.
- Mỗi chunk pháp luật có thể được train thành một LoRA adapter riêng.
- Khi người dùng hỏi, hệ thống retrieve chunk liên quan, chọn LoRA adapter tương ứng, gắn adapter vào LLM rồi generate câu trả lời.

Mặc định pipeline hiện lọc các văn bản **còn hiệu lực** và **liên quan đến an toàn giao thông** từ dataset `th1nhng0/vietnamese-legal-documents`.

Dataset này là snapshot từ `vbpl.vn`; trạng thái hiệu lực phản ánh thời điểm crawl của dataset, không thay thế việc kiểm tra lại trên cổng văn bản chính thức.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
huggingface-cli login
```

`huggingface-cli login` chỉ cần thiết nếu bạn tải model gated trên Hugging Face.

## Chạy Lại Từ Đầu Cho An Toàn Giao Thông

Xóa hoặc ghi đè các artifact cũ tùy nhu cầu. Script bên dưới sẽ ghi lại raw JSON, chunks, retriever và parametric store:

```powershell
python -m legal_prag.data.download_dataset --topic traffic-safety
python -m legal_prag.data.prepare_corpus --chunk-size 900 --chunk-overlap 120
python -m legal_prag.data.build_store
```

Output chính:

```text
data/raw/legal_docs_traffic_safety_active.json
data/processed/chunks.json
artifacts/retriever.joblib
artifacts/parametric_store.joblib
```

Nếu muốn test nhanh với ít văn bản:

```powershell
python -m legal_prag.data.download_dataset --topic traffic-safety --limit 100
python -m legal_prag.data.prepare_corpus --chunk-size 900 --chunk-overlap 120
python -m legal_prag.data.build_store
```

Nếu muốn thêm từ khóa riêng:

```powershell
python -m legal_prag.data.download_dataset --topic traffic-safety --keyword "nồng độ cồn" --keyword "mũ bảo hiểm"
```

Nếu muốn lấy toàn bộ văn bản còn hiệu lực, không lọc theo chủ đề:

```powershell
python -m legal_prag.data.download_dataset --topic all --output data/raw/legal_docs_active.json
python -m legal_prag.data.prepare_corpus --input data/raw/legal_docs_active.json --chunk-size 900 --chunk-overlap 120
python -m legal_prag.data.build_store
```

## Bộ Lọc Dataset

Dataset dùng các config:

- `metadata`: metadata của văn bản, gồm `title`, `nganh`, `linh_vuc`, `tinh_trang_hieu_luc`, `ngay_het_hieu_luc`.
- `content`: nội dung HTML, join với metadata bằng `id`.

Downloader giữ văn bản khi:

```text
tinh_trang_hieu_luc == "Còn hiệu lực"
```

Với `--topic traffic-safety`, script tiếp tục lọc bằng nhóm từ khóa như:

```text
an toàn giao thông
trật tự an toàn giao thông
giao thông đường bộ
đường bộ
đường sắt
đường thủy nội địa
vận tải đường bộ
giấy phép lái xe
đăng kiểm
xử phạt vi phạm giao thông
tai nạn giao thông
```

Bộ lọc xét cả metadata và nội dung văn bản, có normalize Unicode và bỏ dấu khi match.

## Train LoRA Adapters

Train thử vài adapter đầu tiên:

```powershell
python -m legal_prag.training.train_lora_adapters --max-adapters 3 --epochs 1
```

Nếu máy yếu:

```powershell
python -m legal_prag.training.train_lora_adapters --max-adapters 1 --epochs 1 --max-length 512
```

## Chạy Chatbot

Chạy CLI:

```powershell
python -m legal_prag.serving.chatbot
```

Chỉ test retrieval, không tải LLM:

```powershell
python -m legal_prag.serving.chatbot --no-llm
```

## Chạy Web UI

Chạy web demo nhanh, không tải LLM:

```powershell
python -m legal_prag.serving.web_app --no-llm --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Chạy web với PRAG LoRA:

```powershell
python -m legal_prag.serving.web_app --port 8000
```

## Cấu Trúc

```text
legal_prag/
  data/
    download_dataset.py
    prepare_corpus.py
    build_store.py
    merge_traffic_corpus.py
    process_pdf_corpus.py
  retrieval/
    retriever.py
    parametric_store.py
  generation/
    generator.py
    lora_generator.py
  training/
    train_lora_adapters.py
  serving/
    chatbot.py
    web_app.py
  config.py
  utils.py
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

