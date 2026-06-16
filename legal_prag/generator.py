import re

from legal_prag.parametric_store import TemporaryLegalMemory

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 768


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
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {source}",
                    f"So ky hieu: {so_ky_hieu}",
                    f"Ngay ban hanh: {ngay_ban_hanh}",
                    f"Do lien quan: {item.score:.3f}",
                    f"Noi dung: {compact_text(chunk['text'], max_chars=max_chars)}",
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


def build_llama_prompt(question: str, memory: TemporaryLegalMemory) -> str:
    sources = build_sources(memory, max_chars=1100)
    return f"""Dựa vào nguồn sau, trả lời ngắn câu hỏi bằng tiếng Việt.
Chỉ dùng thông tin trong nguồn. Nếu nguồn không có thông tin liên quan, trả lời: "Chưa đủ căn cứ trong nguồn truy xuất."
Tuyệt đối không dùng tiếng Trung, tiếng Nhật, tiếng Hàn, tiếng Anh hoặc ngôn ngữ khác.
Viết đúng 2 dòng:
Trả lời: <1 đến 2 câu ngắn>
Căn cứ: <điều/khoản/mục nếu nguồn có; số hiệu hoặc tên văn bản>

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
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
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

        prompt = build_llama_prompt(question, memory)
        messages = [
            {
                "role": "system",
                "content": "Ban la tro ly phap luat Viet Nam. Chi dua tren nguon duoc cung cap. Tra loi ngan gon, khong lap.",
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
        return add_citation(answer, memory)


def generate_answer(question: str, memory: TemporaryLegalMemory) -> str:
    if not memory.evidence:
        return missing_evidence_answer()

    return summarize_top_evidence(memory)
