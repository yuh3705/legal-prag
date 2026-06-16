import argparse
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from tqdm import tqdm

from legal_prag.config import ACTIVE_STATUS, DATASET_NAME, RAW_JSON, ensure_dirs
from legal_prag.utils import html_to_text, normalize_text, write_json


def is_active_legal_doc(metadata: dict[str, Any]) -> bool:
    status = metadata.get("tinh_trang_hieu_luc")
    return normalize_text(str(status)) == ACTIVE_STATUS


def normalize_row(metadata: dict[str, Any], content_html: str) -> dict[str, Any]:
    doc_id = str(metadata["id"])
    title = metadata.get("title") or metadata.get("so_ky_hieu") or f"van_ban_{doc_id}"
    return {
        "id": doc_id,
        "title": normalize_text(title),
        "text": html_to_text(content_html),
        "metadata": metadata,
    }


def iter_content_rows(dataset_name: str, batch_size: int = 128):
    """Read content parquet directly to avoid datasets large_string -> string cast."""
    parquet_path = hf_hub_download(
        repo_id=dataset_name,
        filename="data/content.parquet",
        repo_type="dataset",
    )
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["id", "content_html"]):
        ids = batch.column("id").to_pylist()
        contents = batch.column("content_html").to_pylist()
        for doc_id, content_html in zip(ids, contents):
            yield {"id": doc_id, "content_html": content_html}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()

    metadata_ds = load_dataset(args.dataset, "metadata", split=args.split, streaming=True)
    active_metadata: dict[str, dict[str, Any]] = {}
    scanned_meta = 0
    for row in tqdm(metadata_ds, desc="Filtering active metadata"):
        scanned_meta += 1
        if not is_active_legal_doc(row):
            continue
        doc_id = str(row["id"])
        active_metadata[doc_id] = dict(row)
        if args.limit and len(active_metadata) >= args.limit:
            break

    docs: list[dict[str, Any]] = []
    for row in tqdm(iter_content_rows(args.dataset), desc="Joining full-text content"):
        doc_id = str(row["id"])
        metadata = active_metadata.get(doc_id)
        if metadata is None:
            continue
        doc = normalize_row(metadata, row.get("content_html") or "")
        if doc["text"]:
            docs.append(doc)
        if len(docs) >= len(active_metadata):
            break

    write_json(RAW_JSON, docs)
    print(f"Scanned metadata rows: {scanned_meta}")
    print(f"Active metadata rows: {len(active_metadata)}")
    print(f"Saved active documents: {len(docs)}")
    print(f"Output: {RAW_JSON}")


if __name__ == "__main__":
    main()
