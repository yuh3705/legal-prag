from legal_prag.config import CHUNKS_JSON, PARAMETRIC_STORE_PATH, RETRIEVER_PATH, ensure_dirs
from legal_prag.parametric_store import ParametricStore
from legal_prag.retriever import LegalRetriever
from legal_prag.utils import read_json


def main() -> None:
    ensure_dirs()
    chunks = read_json(CHUNKS_JSON)
    if not chunks:
        raise RuntimeError("No chunks found. Run prepare_corpus first.")

    retriever = LegalRetriever.fit(chunks)
    retriever.save(str(RETRIEVER_PATH))

    store = ParametricStore.fit(chunks)
    store.attach_chunk_index(chunks)
    store.save(str(PARAMETRIC_STORE_PATH))

    print(f"Chunks: {len(chunks)}")
    print(f"Retriever: {RETRIEVER_PATH}")
    print(f"Parametric store: {PARAMETRIC_STORE_PATH}")


if __name__ == "__main__":
    main()

