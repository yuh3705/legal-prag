from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

RAW_JSON = RAW_DIR / "legal_docs_active.json"
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"
RETRIEVER_PATH = ARTIFACT_DIR / "retriever.joblib"
PARAMETRIC_STORE_PATH = ARTIFACT_DIR / "parametric_store.joblib"
LORA_DIR = ARTIFACT_DIR / "Qwen_Qwen2.5-3B-Instruct" / "lora_adapters"
LORA_MANIFEST_PATH = LORA_DIR / "manifest.json"

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"
ACTIVE_STATUS = "Còn hiệu lực"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
