import re

from legal_prag.retrieval.parametric_store import TemporaryLegalMemory

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512


def compact_text(text: str, max_chars: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def build_sources(memory: TemporaryLegalMemory, max_chars: int = 900) -> str:
    blocks = []
    for idx, item in enumerate(memory.evidence, start=1):
        chunk = item.chunk
        metadata = chunk.get("metadata") or {}
        so_ky_hieu = metadata.get("so_ky_hieu") or metadata.get("so_hieu") or ""
        ngay_ban_hanh = metadata.get("ngay_ban_hanh") or ""
        source = chunk.get("title") or so_ky_hieu or chunk["doc_id"]
        legal_units = chunk.get("_legal_units") or []
        content = " ".join(unit.get("text", "") for unit in legal_units) if legal_units else chunk["text"]
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {source}",
                    f"So ky hieu: {so_ky_hieu}",
                    f"Ngay ban hanh: {ngay_ban_hanh}",
                    f"Do lien quan: {item.score:.3f}",
                    f"Noi dung: {compact_text(content, max_chars=max_chars)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def missing_evidence_answer() -> str:
    return (
        "Toi chua tim thay can cu phu hop trong kho van ban con hieu luc. "
        "Ban nen hoi lai voi tu khoa cu the hon, vi du ten luat, nghi dinh, dieu, khoan."
    )


def build_citation(memory: TemporaryLegalMemory) -> str:
    if not memory.evidence:
        return ""

    chunk = memory.evidence[0].chunk
    metadata = chunk.get("metadata") or {}
    text = chunk.get("text") or ""

    legal_unit = ""
    unit_match = re.search(r"\b(Điều\s+\d+[a-zA-Z]?)\b", text)
    if unit_match:
        legal_unit = unit_match.group(1)
    else:
        numbered_items = re.findall(r"(?:^|[.:;]\s)(\d+)/\s", text)
        if numbered_items:
            legal_unit = f"mục {numbered_items[0]}/"

    so_ky_hieu = metadata.get("so_ky_hieu") or metadata.get("so_hieu") or ""
    loai_van_ban = metadata.get("loai_van_ban") or ""
    title = chunk.get("title") or metadata.get("title") or ""
    ngay_ban_hanh = metadata.get("ngay_ban_hanh") or ""

    document_bits = []
    if loai_van_ban:
        document_bits.append(loai_van_ban)
    if so_ky_hieu:
        document_bits.append(f"số {so_ky_hieu}")
    elif title:
        document_bits.append(title)
    if ngay_ban_hanh:
        document_bits.append(f"ngày {ngay_ban_hanh}")

    citation = ", ".join(bit for bit in [legal_unit, " ".join(document_bits)] if bit)
    return f"Căn cứ: {citation}." if citation else ""


def add_citation(answer: str, memory: TemporaryLegalMemory) -> str:
    answer = (answer or "").strip()
    if "căn cứ:" in answer.lower():
        return answer

    citation = build_citation(memory)
    if not citation:
        return answer
    return f"{answer}\n\n{citation}" if answer else citation


def build_legal_unit_citation(unit: dict, chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    so_ky_hieu = metadata.get("so_ky_hieu") or metadata.get("so_hieu") or ""
    loai_van_ban = metadata.get("loai_van_ban") or ""
    title = chunk.get("title") or metadata.get("title") or ""
    ngay_ban_hanh = metadata.get("ngay_ban_hanh") or ""
    article = unit.get("article") or ""

    document_bits = []
    if loai_van_ban:
        document_bits.append(loai_van_ban)
    if so_ky_hieu:
        document_bits.append(f"số {so_ky_hieu}")
    elif title:
        document_bits.append(title)
    if ngay_ban_hanh:
        document_bits.append(f"ngày {ngay_ban_hanh}")

    citation = ", ".join(bit for bit in [article, " ".join(document_bits)] if bit)
    return f"Căn cứ: {citation}." if citation else ""



def extract_answer_from_legal_units(memory: TemporaryLegalMemory, max_units: int = 3) -> str:
    selected: list[tuple[dict, dict]] = []
    seen = set()
    for item in memory.evidence:
        for unit in item.chunk.get("_legal_units") or []:
            unit_id = unit.get("unit_id") or unit.get("text")
            if unit_id in seen:
                continue
            seen.add(unit_id)
            selected.append((unit, item.chunk))
            if len(selected) >= max_units:
                break
        if len(selected) >= max_units:
            break

    if not selected:
        return ""

    answer = "Theo nguồn truy xuất, cần xử lý như sau:\n" + "\n".join(
        f"- {compact_text(unit.get('text') or '', max_chars=520)}"
        for unit, _ in selected
    )
    citation = build_legal_unit_citation(selected[0][0], selected[0][1])
    return f"{answer}\n\n{citation}" if citation else answer


def summarize_top_evidence(memory: TemporaryLegalMemory, max_chars: int = 1600) -> str:
    item = memory.evidence[0]
    chunk = item.chunk
    metadata = chunk.get("metadata") or {}
    so_ky_hieu = metadata.get("so_ky_hieu") or metadata.get("so_hieu") or ""
    ngay_ban_hanh = metadata.get("ngay_ban_hanh") or ""
    source = chunk.get("title") or so_ky_hieu or chunk["doc_id"]

    summary = compact_text(chunk["text"], max_chars=max_chars)

    source_bits = []
    if so_ky_hieu:
        source_bits.append(so_ky_hieu)
    if ngay_ban_hanh:
        source_bits.append(f"ngay ban hanh {ngay_ban_hanh}")
    source_text = f" ({', '.join(source_bits)})" if source_bits else ""
    return f"{summary}\n\nNguon: {source}{source_text}"


def tokenize_for_answer(text: str) -> set[str]:
    stopwords = {
        "ban",
        "cac",
        "cho",
        "con",
        "cua",
        "các",
        "có",
        "của",
        "khi",
        "mot",
        "một",
        "nha",
        "nhà",
        "những",
        "phai",
        "phải",
        "thi",
        "thì",
        "trong",
        "và",
        "voi",
        "với",
        "xu",
        "xử",
    }
    return {
        token
        for token in re.findall(r"[0-9a-zà-ỹđ]+", (text or "").lower(), flags=re.UNICODE)
        if len(token) > 2 and token not in stopwords
    }


def split_legal_units(text: str) -> list[str]:
    text = compact_text(text, max_chars=6000)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([a-zđ]/)\s*", r" \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(Điều\s+\d+\s*:)\s*", r" \1 ", text, flags=re.IGNORECASE)
    parts = re.split(r"(?=(?:Điều\s+\d+\s*:|[a-zđ]/\s|\d+[./]\s))", text, flags=re.IGNORECASE)
    return [part.strip(" ;.") for part in parts if len(part.strip()) > 40]


def extract_targeted_span(question: str, text: str) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    lower = compact.lower()
    question_lower = (question or "").lower()

    if "xử lý như sau" in lower:
        marker = lower.find("xử lý như sau")
        start = lower.rfind("điều 1", 0, marker)
        if start < 0:
            start = marker
        end_match = re.search(r"\sđiều\s+2\s*[:.]", lower[marker:])
        end = marker + end_match.start() if end_match else len(compact)
        return compact_text(compact[start:end], max_chars=1100)

    asks_ubnd_huyen = "ubnd" in question_lower and (
        "huyện" in question_lower or "huyen" in question_lower
    )
    if asks_ubnd_huyen:
        patterns = [
            r"(\d+[./]\s*UBND\s+các\s+huyện.*?có\s+trách\s+nhiệm\s*:.*?)(?=\s\d+[./]\s|\sĐiều\s+\d+\s*:|$)",
            r"(UBND\s+các\s+huyện.*?có\s+trách\s+nhiệm\s*:.*?)(?=\s\d+[./]\s|\sĐiều\s+\d+\s*:|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if match:
                return compact_text(match.group(1), max_chars=1300)

    return ""


def extract_answer_from_evidence(question: str, memory: TemporaryLegalMemory, max_units: int = 3) -> str:
    if not memory.evidence:
        return missing_evidence_answer()

    legal_unit_answer = extract_answer_from_legal_units(memory, max_units=max_units)
    if legal_unit_answer:
        return legal_unit_answer

    seen_chunk_ids = set()
    for item in memory.evidence:
        chunk_id = item.chunk.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        targeted = extract_targeted_span(question, item.chunk.get("text") or "")
        if targeted:
            return add_citation(
                f"Theo nguồn truy xuất, cần xử lý như sau:\n- {targeted}",
                memory,
            )

    query_terms = tokenize_for_answer(question)
    action_terms = [
        "xử lý",
        "trách nhiệm",
        "phải",
        "cho phép",
        "nghĩa vụ",
        "thực hiện",
        "rà soát",
        "thu hồi",
        "chuyển hợp đồng",
        "đóng góp",
        "kiểm tra",
    ]
    scored: list[tuple[float, str]] = []
    for item in memory.evidence:
        for unit in split_legal_units(item.chunk.get("text") or ""):
            unit_lower = unit.lower()
            score = len(query_terms & tokenize_for_answer(unit))
            score += sum(2 for term in action_terms if term in unit_lower)
            if score > 0:
                scored.append((float(score) + item.score / 1000, unit))

    if not scored:
        return summarize_top_evidence(memory)

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen = set()
    for _, unit in scored:
        normalized = unit.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(compact_text(unit, max_chars=520))
        if len(selected) >= max_units:
            break

    answer = "Theo nguồn truy xuất, cần xử lý như sau:\n" + "\n".join(
        f"- {unit}" for unit in selected
    )
    return add_citation(answer, memory)


def is_too_vague_answer(answer: str) -> bool:
    text = " ".join((answer or "").lower().split())
    if any(marker in text for marker in ["chưa đủ căn cứ", "chua du can cu", "không rõ", "khong ro"]):
        return True
    if len(text) < 120 and re.search(r"\b(theo|căn cứ|quyết định|chỉ thị|nghị định)\b", text):
        return True
    vague_patterns = [
        r"xử lý theo (quyết định|chỉ thị|nghị định)",
        r"phải làm theo (quyết định|chỉ thị|nghị định)",
    ]
    return any(re.search(pattern, text) for pattern in vague_patterns)


def build_prompt(question: str, memory: TemporaryLegalMemory, include_sources: bool = True) -> str:
    if not include_sources:
        return f"""Trả lời câu hỏi pháp luật Việt Nam bằng tiếng Việt rõ ràng và trực tiếp.
Dựa trên tri thức đã được nạp trong mô hình/adapter đang được kích hoạt. Không bịa nếu không đủ căn cứ.
Nếu không chắc chắn, trả lời: "Chưa đủ căn cứ để trả lời chắc chắn."
Viết ngắn gọn. Nếu biết căn cứ pháp lý thì ghi một dòng căn cứ.
Tuyệt đối không dùng tiếng Trung, tiếng Nhật, tiếng Hàn, tiếng Anh hoặc ngôn ngữ khác.

Câu hỏi: {question}

Trả lời:
"""

    sources = build_sources(memory, max_chars=1100)
    return f"""Đưa vào nguồn sau, trả lời câu hỏi pháp luật Việt Nam rõ ràng và đúng trọng tâm.
Chỉ dùng thông tin trong nguồn. Nếu nguồn không có thông tin liên quan, trả lời: "Chưa đủ căn cứ trong nguồn truy xuất."

Nguồn:
{sources}

Câu hỏi: {question}

Trả lời:
"""


def generation_kwargs(tokenizer, max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


class LlamaGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        include_sources: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.include_sources = include_sources
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    def generate(self, question: str, memory: TemporaryLegalMemory) -> str:
        if not memory.evidence:
            return missing_evidence_answer()

        prompt = build_prompt(question, memory, include_sources=self.include_sources)
        system_content = (
            "Ban la tro ly phap luat Viet Nam. Chi dua tren nguon duoc cung cap. Tra loi ro truong hop ap dung, muc phat va can cu; khong lap."
            if self.include_sources
            else "Ban la tro ly phap luat Viet Nam. Tra loi dua tren tri thuc da duoc nap trong mo hinh, ngan gon va khong bia."
        )
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs(self.tokenizer, self.max_new_tokens),
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if self.include_sources and is_too_vague_answer(answer):
            return extract_answer_from_evidence(question, memory)
        return add_citation(answer, memory) if self.include_sources else answer


def generate_answer(question: str, memory: TemporaryLegalMemory) -> str:
    if not memory.evidence:
        return missing_evidence_answer()

    return extract_answer_from_evidence(question, memory)
