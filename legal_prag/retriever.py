from dataclasses import dataclass
import re
from typing import Any

import joblib
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    score: float
    chunk: dict[str, Any]


class LegalRetriever:
    def __init__(self, bm25: BM25Okapi, tokenized_corpus: list[list[str]], chunks: list[dict[str, Any]]):
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.chunks = chunks

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = (text or "").lower()
        return re.findall(r"[0-9a-zà-ỹđ]+", text, flags=re.UNICODE)

    @classmethod
    def fit(cls, chunks: list[dict[str, Any]]) -> "LegalRetriever":
        texts = [f"{c['title']} {c['text']}" for c in chunks]
        tokenized_corpus = [cls.tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25, tokenized_corpus, chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(query_tokens))
        if not np.any(scores):
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(score=float(scores[i]), chunk=self.chunks[int(i)])
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LegalRetriever":
        return joblib.load(path)
