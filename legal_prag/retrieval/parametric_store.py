from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from legal_prag.retrieval.retriever import RetrievedChunk


@dataclass
class TemporaryLegalMemory:
    query_terms: list[str]
    evidence: list[RetrievedChunk]


class ParametricStore:
    """Lightweight document parameter store.

    PRAG gốc dùng LoRA parameters cho từng document. Bản demo này dùng TF-IDF
    feature weights như document parameters để có thể chạy nhanh trên CPU.
    """

    def __init__(self, vectorizer: TfidfVectorizer, doc_params: Any, feature_names: np.ndarray):
        self.vectorizer = vectorizer
        self.doc_params = doc_params
        self.feature_names = feature_names

    @classmethod
    def fit(cls, chunks: list[dict[str, Any]]) -> "ParametricStore":
        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
        )
        doc_params = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        return cls(vectorizer, doc_params, feature_names)

    def update(self, retrieved: list[RetrievedChunk], top_terms: int = 12) -> TemporaryLegalMemory:
        if not retrieved:
            return TemporaryLegalMemory(query_terms=[], evidence=[])

        row_indices = []
        weights = []
        chunk_ids = {item.chunk["chunk_id"]: item for item in retrieved}
        chunk_id_to_index = getattr(self, "chunk_id_to_index", None)
        if chunk_id_to_index is None:
            raise RuntimeError("ParametricStore is missing chunk_id_to_index. Rebuild the store.")

        for item in retrieved:
            idx = chunk_id_to_index.get(item.chunk["chunk_id"])
            if idx is not None:
                row_indices.append(idx)
                weights.append(max(item.score, 0.001))

        if not row_indices:
            return TemporaryLegalMemory(query_terms=[], evidence=list(chunk_ids.values()))

        selected = self.doc_params[row_indices]
        merged = np.asarray(selected.multiply(np.array(weights)[:, None]).sum(axis=0)).ravel()
        if not np.any(merged):
            return TemporaryLegalMemory(query_terms=[], evidence=list(chunk_ids.values()))

        top_indices = np.argsort(merged)[::-1][:top_terms]
        terms = [str(self.feature_names[i]) for i in top_indices if merged[i] > 0]
        return TemporaryLegalMemory(query_terms=terms, evidence=list(chunk_ids.values()))

    def attach_chunk_index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunk_id_to_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(chunks)}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ParametricStore":
        return joblib.load(path)
