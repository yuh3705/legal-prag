import argparse
from typing import Any

from tqdm import tqdm

from legal_prag.config import CHUNKS_JSON, RAW_JSON, ensure_dirs
from legal_prag.utils import normalize_text, read_json, write_json


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = normalize_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(docs: list[dict[str, Any]], chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in tqdm(docs, desc="Chunking documents"):
        for chunk_id, text in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}::{chunk_id}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": text,
                    "metadata": doc.get("metadata") or {},
                }
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_JSON))
    parser.add_argument("--output", default=str(CHUNKS_JSON))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    ensure_dirs()
    docs = read_json(args.input)
    chunks = build_chunks(docs, args.chunk_size, args.chunk_overlap)
    write_json(args.output, chunks)
    print(f"Saved chunks: {len(chunks)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
