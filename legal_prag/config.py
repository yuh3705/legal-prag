from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

RAW_JSON = RAW_DIR / "legal_docs_traffic_safety_active.json"
RAW_JSON_FALLBACKS = [
    RAW_DIR / "legal_docs_traffic_safety_road_all.json",
    RAW_DIR / "legal_docs_traffic.json",
    RAW_DIR / "legal_docs_active.json",
]
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"
RETRIEVER_PATH = ARTIFACT_DIR / "retriever.joblib"
PARAMETRIC_STORE_PATH = ARTIFACT_DIR / "parametric_store.joblib"
LORA_DIR = ARTIFACT_DIR / "Qwen_Qwen2.5-3B-Instruct" / "lora_adapters"
LORA_MANIFEST_PATH = LORA_DIR / "manifest.json"
EMBEDDING_MODEL_NAME = "mainguyen9/vietlegal-harrier-0.6b"
EMBEDDING_QUERY_INSTRUCTION = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\n"
    "Query: "
)

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"
ACTIVE_STATUS = "Còn hiệu lực"

TRAFFIC_SAFETY_KEYWORDS = [
    "an toàn giao thông",
    "trật tự, an toàn giao thông",
    "trật tự an toàn giao thông",
    "bảo đảm trật tự an toàn giao thông",
    "giao thông đường bộ",
    "luật giao thông đường bộ",
    "luật trật tự, an toàn giao thông đường bộ",
    "đường bộ cao tốc",
    "báo hiệu đường bộ",
    "phương tiện cơ giới đường bộ",
    "vận tải đường bộ",
    "kinh doanh vận tải bằng xe ô tô",
    "xe ô tô",
    "xe mô tô",
    "xe gắn máy",
    "người lái xe",
    "đào tạo lái xe",
    "sát hạch lái xe",
    "giấy phép lái xe",
    "đăng ký xe",
    "đăng kiểm",
    "kiểm định an toàn kỹ thuật",
    "tuần tra, kiểm soát",
    "kiểm soát giao thông",
    "xử phạt vi phạm giao thông",
    "xử phạt vi phạm hành chính trong lĩnh vực giao thông",
    "vi phạm giao thông",
    "tai nạn giao thông",
    "nồng độ cồn",
    "mũ bảo hiểm",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
