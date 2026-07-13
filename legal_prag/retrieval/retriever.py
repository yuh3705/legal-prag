from collections import OrderedDict
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

import joblib
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    score: float
    chunk: dict[str, Any]


@dataclass
class LegalUnit:
    unit_id: str
    chunk_id: str
    doc_id: str
    title: str
    text: str
    article: str
    clause: str
    point: str
    vehicle_group: str
    violation_types: list[str]
    fine_min: int | None
    fine_max: int | None
    is_fine_unit: bool
    is_penalty_document: bool


class LegalRetriever:
    def __init__(
        self,
        bm25: BM25Okapi,
        tokenized_corpus: list[list[str]],
        chunks: list[dict[str, Any]],
        dense_embeddings: np.ndarray | None = None,
        embedding_model_name: str | None = None,
        dense_weight: float = 0.65,
        legal_units: list[dict[str, Any]] | None = None,
    ):
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.chunks = chunks
        self.dense_embeddings = dense_embeddings
        self.embedding_model_name = embedding_model_name
        self.dense_weight = dense_weight
        self.legal_units = legal_units or []
        self.unit_bm25 = BM25Okapi([self.tokenize(unit["text"]) for unit in self.legal_units]) if self.legal_units else None
        self.chunk_id_to_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(chunks)}
        self.enable_dense = True
        self.dense_device = "auto"
        self._embedding_model = None
        self._query_embedding_cache = OrderedDict()
        self._query_embedding_cache_size = 64

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = (text or "").lower()
        return re.findall(r"[0-9a-zà-ỹđ]+", text, flags=re.UNICODE)

    @staticmethod
    def strip_vietnamese(text: str) -> str:
        text = unicodedata.normalize("NFD", text or "")
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.replace("đ", "d").replace("Đ", "D").lower()

    @staticmethod
    def contains_phrase(text: str, phrase: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None

    @classmethod
    def contains_any_phrase(cls, text: str, phrases: list[str]) -> bool:
        return any(cls.contains_phrase(text, phrase) for phrase in phrases)

    @staticmethod
    def parse_money(value: str) -> int | None:
        digits = re.sub(r"\D", "", value or "")
        return int(digits) if digits else None

    @staticmethod
    def chunk_position(chunk_id: str | None) -> tuple[str, int] | None:
        if not chunk_id or "::" not in chunk_id:
            return None
        doc_id, index = chunk_id.rsplit("::", 1)
        if not index.isdigit():
            return None
        return doc_id, int(index)

    @classmethod
    def detect_vehicle_group(cls, text: str) -> str:
        normalized = cls.strip_vietnamese(text)
        groups = [
            ("special_machine", ["xe may chuyen dung", "may chuyen dung"]),
            ("motorbike", ["xe mo to", "mo to", "moto", "xe may", "gan may"]),
            ("car", ["xe o to", "o to", "oto", "xe con", "xe tai", "xe khach", "bon banh"]),
            ("bicycle", ["xe dap", "xe tho so", "xe xich lo"]),
            ("pedestrian", ["nguoi di bo", "di bo"]),
        ]
        hits: list[tuple[int, str]] = []
        for group, phrases in groups:
            positions = [normalized.find(phrase) for phrase in phrases if cls.contains_phrase(normalized, phrase)]
            if positions:
                hits.append((min(positions), group))
        if not hits:
            return ""
        hits.sort(key=lambda item: item[0])
        return hits[0][1]

    @classmethod
    def detect_violation_types(cls, text: str) -> list[str]:
        normalized = cls.strip_vietnamese(text)
        types = []
        if "den tin hieu" in normalized or "den do" in normalized or "hieu lenh cua den" in normalized:
            types.append("traffic_light")
        if "di khong dung phan duong" in normalized or "khong dung lan duong" in normalized or "trai lan" in normalized:
            types.append("wrong_lane")
        if "duong cao toc" in normalized:
            types.append("highway")
        if "nong do con" in normalized or "hoi tho" in normalized:
            types.append("alcohol")
        if "toc do" in normalized or "qua toc do" in normalized:
            types.append("speed")
        if "mu bao hiem" in normalized:
            types.append("helmet")
        if "di nguoc chieu" in normalized or "cam di nguoc chieu" in normalized:
            types.append("wrong_way")
        return types

    @classmethod
    def query_features(cls, query: str) -> dict[str, Any]:
        normalized = cls.strip_vietnamese(query)
        return {
            "vehicle_group": cls.detect_vehicle_group(query),
            "violation_types": cls.detect_violation_types(query),
            "asks_fine": any(term in normalized for term in ["phat", "muc phat", "bao nhieu tien"]),
            "asks_penalty": any(term in normalized for term in ["phat", "xu phat", "muc phat"]),
        }

    @classmethod
    def split_articles(cls, text: str) -> list[tuple[str, str, str]]:
        compact = re.sub(r"\s+", " ", text or "").strip()
        article_pattern = re.compile(r"(Điều\s+(\d+[a-zA-Z]?)\.\s+.*?)(?=\s+Điều\s+\d+[a-zA-Z]?\.\s+|$)", re.DOTALL)
        articles: list[tuple[str, str, str]] = []
        for match in article_pattern.finditer(compact):
            article_text = match.group(1).strip()
            article_no = match.group(2)
            title_end = re.search(r"\s+1\.\s+", article_text)
            title = article_text[: title_end.start()].strip() if title_end else article_text[:300].strip()
            articles.append((article_no, title, article_text))
        return articles

    @classmethod
    def split_fine_clauses(cls, article_text: str) -> list[tuple[str, str]]:
        clause_pattern = re.compile(
            r"(\d+)\.\s*Phạt\s+(?:tiền|liền)\s+từ\s+.*?(?=\s+\d+\.\s+(?:Phạt|Ngoài|Thực|Trường|Hình|Biện)|$)",
            re.IGNORECASE | re.DOTALL,
        )
        return [(match.group(1), match.group(0).strip()) for match in clause_pattern.finditer(article_text)]

    @classmethod
    def split_points(cls, clause_text: str) -> list[tuple[str, str]]:
        marker = re.search(r"sau\s+đây\s*:\s*", clause_text, flags=re.IGNORECASE)
        body = clause_text[marker.end() :] if marker else clause_text
        points = [
            (match.group(1).lower(), match.group(2).strip(" ;."))
            for match in re.finditer(r"([a-zđ])\)\s*(.*?)(?=\s+[a-zđ]\)\s|$)", body, flags=re.IGNORECASE | re.DOTALL)
        ]
        return points or [("", clause_text.strip(" ;."))]

    @classmethod
    def extract_fine_range(cls, text: str) -> tuple[int | None, int | None]:
        match = re.search(r"từ\s+([\d.]+)\s+đồng\s+đến\s+([\d.]+)\s+đồng", text, flags=re.IGNORECASE)
        if not match:
            return None, None
        return cls.parse_money(match.group(1)), cls.parse_money(match.group(2))

    @classmethod
    def map_unit_to_chunk(cls, unit_text: str, doc_id: str, chunks: list[dict[str, Any]]) -> str:
        normalized_unit = cls.strip_vietnamese(unit_text)
        unit_tokens = {
            token
            for token in re.findall(r"[0-9a-z]+", normalized_unit)
            if len(token) >= 3 and token not in {"phat", "tien", "dong", "nguoi", "dieu", "khien", "thuc", "hien"}
        }
        best_score = -1.0
        best_chunk_id = ""
        fragments = [
            fragment.strip()
            for fragment in re.split(r"[;:]", normalized_unit)
            if len(fragment.strip()) >= 35
        ]
        for chunk in chunks:
            if str(chunk.get("doc_id")) != str(doc_id):
                continue
            chunk_text = cls.strip_vietnamese(chunk.get("text") or "")
            score = 0.0
            if normalized_unit[:120] and normalized_unit[:120] in chunk_text:
                score += 80
            if normalized_unit[-120:] and normalized_unit[-120:] in chunk_text:
                score += 60
            for fragment in fragments:
                if fragment in chunk_text:
                    score += 120
            chunk_tokens = set(re.findall(r"[0-9a-z]+", chunk_text))
            score += len(unit_tokens & chunk_tokens)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.get("chunk_id", "")
        return best_chunk_id

    @classmethod
    def build_legal_units(cls, raw_docs: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for doc in raw_docs:
            doc_id = str(doc.get("id", ""))
            title = str(doc.get("title", ""))
            text = doc.get("text") or ""
            is_penalty_document = any(
                phrase in cls.strip_vietnamese(f"{title} {text[:1000]}")
                for phrase in ["xu phat", "vi pham hanh chinh", "nghi dinh"]
            )
            for article_no, article_title, article_text in cls.split_articles(text):
                article_vehicle_group = cls.detect_vehicle_group(article_title)
                for clause_no, clause_text in cls.split_fine_clauses(article_text):
                    fine_min, fine_max = cls.extract_fine_range(clause_text)
                    fine_intro = clause_text.split("sau đây", 1)[0].strip(" :;.")
                    for point, point_text in cls.split_points(clause_text):
                        unit_text = f"{fine_intro}; {point_text}" if point else clause_text
                        unit_vehicle_group = cls.detect_vehicle_group(f"{article_title} {fine_intro} {point_text}") or article_vehicle_group
                        chunk_id = cls.map_unit_to_chunk(unit_text, doc_id, chunks)
                        if not chunk_id:
                            continue
                        units.append(
                            {
                                "unit_id": f"{doc_id}::d{article_no}::k{clause_no}::{point or 'unit'}",
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "title": title,
                                "text": unit_text,
                                "article": f"Điều {article_no}",
                                "clause": clause_no,
                                "point": point,
                                "vehicle_group": unit_vehicle_group,
                                "violation_types": cls.detect_violation_types(unit_text),
                                "fine_min": fine_min,
                                "fine_max": fine_max,
                                "is_fine_unit": fine_min is not None and fine_max is not None,
                                "is_penalty_document": is_penalty_document,
                            }
                        )
        return units

    @classmethod
    def fit(
        cls,
        chunks: list[dict[str, Any]],
        embedding_model_name: str | None = None,
        dense_batch_size: int = 8,
        raw_docs: list[dict[str, Any]] | None = None,
    ) -> "LegalRetriever":
        texts = [f"{c['title']} {c['text']}" for c in chunks]
        tokenized_corpus = [cls.tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        dense_embeddings = None
        if embedding_model_name:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model_name)
            dense_embeddings = model.encode(
                texts,
                batch_size=dense_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
        legal_units = cls.build_legal_units(raw_docs or [], chunks)
        return cls(bm25, tokenized_corpus, chunks, dense_embeddings, embedding_model_name, legal_units=legal_units)

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0 or not np.any(scores):
            return np.zeros_like(scores, dtype="float32")
        max_score = float(scores.max())
        min_score = float(scores.min())
        if max_score == min_score:
            return np.ones_like(scores, dtype="float32")
        return ((scores - min_score) / (max_score - min_score)).astype("float32")

    def has_dense_index(self) -> bool:
        embeddings = getattr(self, "dense_embeddings", None)
        model_name = getattr(self, "embedding_model_name", None)
        return embeddings is not None and model_name is not None

    def set_dense_enabled(self, enabled: bool) -> None:
        self.enable_dense = enabled

    def set_dense_device(self, device: str | None) -> None:
        normalized = (device or "auto").strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("dense device must be one of: auto, cpu, cuda")
        if getattr(self, "dense_device", "auto") != normalized:
            self._embedding_model = None
            if getattr(self, "_query_embedding_cache", None) is None:
                self._query_embedding_cache = OrderedDict()
            else:
                self._query_embedding_cache.clear()
        self.dense_device = normalized

    def resolve_dense_device(self) -> str | None:
        device = getattr(self, "dense_device", "auto")
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return None

    def load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            device = self.resolve_dense_device()
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        return self._embedding_model

    def encode_dense_query(self, query: str) -> np.ndarray:
        cache = getattr(self, "_query_embedding_cache", None)
        if cache is None:
            self._query_embedding_cache = OrderedDict()
            cache = self._query_embedding_cache

        formatted_query = self.format_dense_query(query)
        cached = cache.get(formatted_query)
        if cached is not None:
            cache.move_to_end(formatted_query)
            return cached

        model = self.load_embedding_model()
        embedding = model.encode(
            [formatted_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]
        cache[formatted_query] = embedding
        max_size = getattr(self, "_query_embedding_cache_size", 64)
        while len(cache) > max_size:
            cache.popitem(last=False)
        return embedding

    def warmup_dense(self) -> None:
        if getattr(self, "enable_dense", True) and self.has_dense_index():
            self.encode_dense_query("câu hỏi pháp luật giao thông đường bộ")

    def format_dense_query(self, query: str) -> str:
        try:
            from legal_prag.config import EMBEDDING_QUERY_INSTRUCTION
        except Exception:
            EMBEDDING_QUERY_INSTRUCTION = (
                "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\n"
                "Query: "
            )
        return f"{EMBEDDING_QUERY_INSTRUCTION}{query}"

    def is_signal_fine_query(self, query: str) -> bool:
        normalized = self.strip_vietnamese(query)
        asks_signal = (
            "den do" in normalized
            or "den tin hieu" in normalized
            or "hieu lenh den" in normalized
        )
        asks_fine = (
            "phat" in normalized
            or "vuot" in normalized
            or "khong chap hanh" in normalized
            or "khong tuan thu" in normalized
        )
        return asks_signal and asks_fine

    def query_vehicle_group(self, query: str) -> str:
        normalized = self.strip_vietnamese(query)
        groups = [
            ("special_machine", ["may chuyen dung"]),
            ("motorbike", ["xe may", "mo to", "moto", "gan may"]),
            ("car", ["o to", "oto", "xe oto", "xe o to", "xe con", "xe tai", "xe khach"]),
            ("bicycle", ["xe dap", "xe tho so"]),
            ("pedestrian", ["nguoi di bo", "di bo"]),
        ]
        hits: list[tuple[int, str]] = []
        for group, terms in groups:
            positions = [normalized.find(term) for term in terms if self.contains_phrase(normalized, term)]
            if positions:
                hits.append((min(positions), group))
        if not hits:
            return ""
        hits.sort(key=lambda hit: hit[0])
        return hits[0][1]

    def text_matches_vehicle_group(self, text: str, group: str) -> bool:
        normalized = self.strip_vietnamese(text)
        if group == "motorbike":
            return self.contains_any_phrase(normalized, ["xe may", "mo to", "moto", "gan may"])
        if group == "special_machine":
            return self.contains_phrase(normalized, "may chuyen dung")
        if group == "car":
            return self.contains_any_phrase(
                normalized,
                ["o to", "oto", "xe oto", "xe o to"],
            )
        if group == "bicycle":
            return self.contains_any_phrase(normalized, ["xe dap", "xe tho so", "tho so"])
        if group == "pedestrian":
            return self.contains_any_phrase(normalized, ["nguoi di bo", "di bo"])
        return False

    def text_conflicts_vehicle_group(self, text: str, group: str) -> bool:
        normalized = self.strip_vietnamese(text)
        conflicts = {
            "motorbike": ["may chuyen dung", "nguoi di bo", "xe dap", "xe tho so"],
            "special_machine": ["nguoi di bo", "xe dap", "xe tho so", "mo to", "gan may", "xe may"],
            "car": ["nguoi di bo", "xe dap", "xe tho so", "mo to", "gan may", "xe may", "may chuyen dung"],
            "bicycle": ["nguoi di bo", "mo to", "gan may", "xe may", "may chuyen dung", "o to", "oto"],
            "pedestrian": ["xe dap", "mo to", "gan may", "xe may", "may chuyen dung", "o to", "oto"],
        }
        return self.contains_any_phrase(normalized, conflicts.get(group, []))

    def chunk_search_features(self) -> tuple[list[str], list[tuple[str, int] | None]]:
        if getattr(self, "_normalized_chunk_texts", None) is None:
            self._normalized_chunk_texts = [
                self.strip_vietnamese(f"{chunk.get('title', '')} {chunk.get('text', '')}")
                for chunk in self.chunks
            ]
        if getattr(self, "_chunk_positions", None) is None:
            self._chunk_positions = [self.chunk_position(chunk.get("chunk_id")) for chunk in self.chunks]
        return self._normalized_chunk_texts, self._chunk_positions

    def nearby_vehicle_score(
        self,
        chunk_index: int,
        group: str,
        chunk_texts: list[str],
        positions: list[tuple[str, int] | None],
    ) -> float:
        position = positions[chunk_index]
        if not position:
            return 0.0
        doc_id, index = position
        best_match = 0.0
        worst_conflict = 0.0
        for other_index, other_position in enumerate(positions):
            if not other_position or other_position[0] != doc_id:
                continue
            distance = abs(index - other_position[1])
            if distance > 3:
                continue
            other_text = chunk_texts[other_index]
            if self.text_matches_vehicle_group(other_text, group):
                best_match = max(best_match, 45.0 - distance * 10.0)
            elif self.text_conflicts_vehicle_group(other_text, group):
                worst_conflict = min(worst_conflict, -25.0 + distance * 5.0)
        return best_match if best_match > 0 else worst_conflict

    def signal_fine_boost_scores(self, query: str) -> np.ndarray:
        boosts = np.zeros(len(self.chunks), dtype="float32")
        if not self.is_signal_fine_query(query):
            return boosts

        group = self.query_vehicle_group(query)
        chunk_texts, positions = self.chunk_search_features()
        for idx, text in enumerate(chunk_texts):
            has_signal_violation = "khong chap hanh" in text and "den tin hieu" in text
            has_signal_context = has_signal_violation or "den tin hieu" in text
            if not group:
                if has_signal_violation:
                    boosts[idx] += 100
                elif has_signal_context:
                    boosts[idx] += 35
                continue

            vehicle_score = 0.0
            direct_match = self.text_matches_vehicle_group(text, group)
            conflicts = self.text_conflicts_vehicle_group(text, group) and not direct_match
            if direct_match:
                vehicle_score += 55
            else:
                nearby_score = self.nearby_vehicle_score(idx, group, chunk_texts, positions)
                if conflicts and nearby_score <= 0:
                    vehicle_score -= 85
                vehicle_score += nearby_score

            if has_signal_violation:
                boosts[idx] += (120 if vehicle_score > 0 else 15) + vehicle_score
            elif has_signal_context:
                boosts[idx] += (45 if vehicle_score > 0 else 5) + vehicle_score
            else:
                boosts[idx] += vehicle_score

        return boosts

    def ensure_legal_unit_index(self) -> None:
        if getattr(self, "chunk_id_to_index", None) is None:
            self.chunk_id_to_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(self.chunks)}
        if getattr(self, "legal_units", None) is None:
            self.legal_units = []
        if getattr(self, "unit_bm25", None) is None and self.legal_units:
            self.unit_bm25 = BM25Okapi([self.tokenize(unit["text"]) for unit in self.legal_units])

    def attach_legal_units(self, raw_docs: list[dict[str, Any]]) -> None:
        self.legal_units = self.build_legal_units(raw_docs, self.chunks)
        self.unit_bm25 = BM25Okapi([self.tokenize(unit["text"]) for unit in self.legal_units]) if self.legal_units else None
        self.chunk_id_to_index = {chunk["chunk_id"]: idx for idx, chunk in enumerate(self.chunks)}

    def legal_unit_boost(self, query_features: dict[str, Any], unit: dict[str, Any]) -> float:
        boost = 0.0
        query_vehicle = query_features.get("vehicle_group") or ""
        unit_vehicle = unit.get("vehicle_group") or ""
        if query_vehicle:
            if unit_vehicle == query_vehicle:
                boost += 90
            elif unit_vehicle:
                boost -= 80

        query_violations = set(query_features.get("violation_types") or [])
        unit_violations = set(unit.get("violation_types") or [])
        if query_violations:
            if query_violations & unit_violations:
                boost += 130
            elif unit_violations:
                boost -= 20

        if query_features.get("asks_fine") and unit.get("is_fine_unit"):
            boost += 55
        if query_features.get("asks_penalty") and unit.get("is_penalty_document"):
            boost += 35
        return boost

    def legal_unit_scores(self, query: str) -> np.ndarray:
        self.ensure_legal_unit_index()
        if not getattr(self, "legal_units", None) or getattr(self, "unit_bm25", None) is None:
            return np.zeros(len(self.chunks), dtype="float32")

        features = self.query_features(query)
        if not (features.get("asks_penalty") or features.get("vehicle_group") or features.get("violation_types")):
            return np.zeros(len(self.chunks), dtype="float32")

        unit_scores = np.asarray(self.unit_bm25.get_scores(self.tokenize(query)), dtype="float32")
        for idx, unit in enumerate(self.legal_units):
            unit_scores[idx] += self.legal_unit_boost(features, unit)

        chunk_scores = np.zeros(len(self.chunks), dtype="float32")
        self._last_legal_unit_matches = {}
        if not np.any(unit_scores):
            return chunk_scores

        top_unit_count = min(len(unit_scores), 80)
        for unit_idx in np.argsort(unit_scores)[::-1][:top_unit_count]:
            score = float(unit_scores[unit_idx])
            if score <= 0:
                continue
            chunk_id = self.legal_units[int(unit_idx)].get("chunk_id")
            chunk_idx = self.chunk_id_to_index.get(chunk_id)
            if chunk_idx is not None:
                chunk_scores[chunk_idx] = max(chunk_scores[chunk_idx], score)
                self._last_legal_unit_matches.setdefault(chunk_id, []).append(self.legal_units[int(unit_idx)])
        return chunk_scores

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float32")
        bm25_scores = bm25_scores + self.signal_fine_boost_scores(query)
        scores = bm25_scores
        if getattr(self, "enable_dense", True) and self.has_dense_index():
            try:
                query_embedding = self.encode_dense_query(query)
                dense_scores = np.asarray(self.dense_embeddings @ query_embedding, dtype="float32")
                scores = (
                    (1 - float(getattr(self, "dense_weight", 0.65))) * self.normalize_scores(bm25_scores)
                    + float(getattr(self, "dense_weight", 0.65)) * self.normalize_scores(dense_scores)
                )
            except (ImportError, OSError):
                scores = bm25_scores
        if not np.any(scores):
            return []

        unit_scores = self.legal_unit_scores(query)
        pool_k = min(len(scores), max(50, top_k * 8))
        candidate_indices = set(int(i) for i in np.argsort(scores)[::-1][:pool_k] if scores[i] > 0)
        if np.any(unit_scores):
            candidate_indices.update(int(i) for i in np.argsort(unit_scores)[::-1][:pool_k] if unit_scores[i] > 0)
            final_scores = (
                0.65 * self.normalize_scores(scores)
                + 1.35 * self.normalize_scores(unit_scores)
            )
        else:
            final_scores = scores

        top_indices = sorted(candidate_indices, key=lambda idx: float(final_scores[idx]), reverse=True)[:top_k]
        results = []
        unit_matches = getattr(self, "_last_legal_unit_matches", {})
        for i in top_indices:
            if final_scores[i] <= 0:
                continue
            chunk = dict(self.chunks[int(i)])
            matches = unit_matches.get(chunk.get("chunk_id"), [])
            if matches:
                chunk["_legal_units"] = matches[:3]
            results.append(RetrievedChunk(score=float(final_scores[i]), chunk=chunk))
        return results

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LegalRetriever":
        return joblib.load(path)
