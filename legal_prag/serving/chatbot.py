import argparse
import unicodedata
from typing import Any

from legal_prag.config import PARAMETRIC_STORE_PATH, RETRIEVER_PATH
from legal_prag.generation.generator import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_NAME, LlamaGenerator, generate_answer
from legal_prag.generation.lora_generator import LoraPragGenerator
from legal_prag.retrieval.parametric_store import ParametricStore
from legal_prag.retrieval.retriever import LegalRetriever

GENERATION_MODES = {"rag", "prag", "hybrid"}


def strip_vietnamese(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D").lower()


def last_user_question(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""

    for item in reversed(history):
        role = str(item.get("role", "")).strip().lower()
        if role == "user":
            return " ".join(str(item.get("text", "")).split())
    return ""


def is_vague_follow_up(question: str) -> bool:
    normalized = strip_vietnamese(question)
    tokens = normalized.split()
    if len(tokens) > 6:
        return False
    vague_markers = [
        "nhu nao",
        "sao",
        "la sao",
        "noi ro",
        "cu the",
        "y la",
    ]
    return any(marker in normalized for marker in vague_markers)


def has_vehicle_reference(question: str) -> bool:
    normalized = strip_vietnamese(question)
    vehicle_markers = [
        "o to",
        "xe oto",
        "xe may",
        "mo to",
        "moto",
        "gan may",
        "xe dap",
        "xe tho so",
        "di bo",
        "nguoi di bo",
        "may chuyen dung",
    ]
    return any(marker in normalized for marker in vehicle_markers)


def expand_legal_query(question: str) -> str:
    """Add legal synonyms that users naturally ask with different wording."""
    normalized = strip_vietnamese(question)
    expansions: list[str] = []

    if "oto" in normalized or "xe oto" in normalized:
        expansions.append("ô tô xe ô tô")
    if "xe may" in normalized:
        expansions.append("mô tô xe gắn máy")
    if "khong tuan thu" in normalized:
        expansions.append("không chấp hành")
    if (
        ("den do" in normalized or "den tin hieu" in normalized or "hieu lenh den" in normalized)
        and ("vuot" in normalized or "khong chap hanh" in normalized or "khong tuan thu" in normalized)
    ):
        expansions.append("không chấp hành hiệu lệnh của đèn tín hiệu giao thông")

    if not expansions:
        return question
    return question + "\n" + " ".join(expansions)


def contextual_question(question: str, history: list[dict[str, Any]] | None = None) -> str:
    previous_question = last_user_question(history)
    if not previous_question or not is_vague_follow_up(question):
        return question
    if has_vehicle_reference(question):
        return f"{question}\nCùng hành vi trong câu trước: {previous_question}"
    return f"{previous_question}\nYêu cầu làm rõ: {question}"


class LegalPragChatbot:
    def __init__(
        self,
        top_k: int = 8,
        use_llm: bool = True,
        use_lora: bool = True,
        use_dense: bool = True,
        dense_device: str | None = None,
        mode: str = "hybrid",
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        top_n_adapters: int = 3,
    ):
        if mode not in GENERATION_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(GENERATION_MODES))}")
        if not use_lora:
            mode = "rag"
        self.top_k = top_k
        self.mode = mode
        self.retriever = LegalRetriever.load(str(RETRIEVER_PATH))
        self.retriever.set_dense_enabled(use_dense)
        self.retriever.set_dense_device(dense_device)
        self.store = ParametricStore.load(str(PARAMETRIC_STORE_PATH))
        if not use_llm:
            self.generator = None
        elif mode == "rag":
            self.generator = LlamaGenerator(model_name, max_new_tokens, include_sources=True)
        elif mode == "prag":
            self.generator = LoraPragGenerator(
                model_name,
                max_new_tokens,
                include_sources=False,
                top_n_adapters=top_n_adapters,
            )
        elif mode == "hybrid":
            self.generator = LoraPragGenerator(
                model_name,
                max_new_tokens,
                include_sources=True,
                top_n_adapters=top_n_adapters,
            )

    def warmup(self) -> None:
        self.retriever.warmup_dense()

    def answer(self, question: str, history: list[dict[str, Any]] | None = None) -> str:
        question_with_context = contextual_question(question, history)
        retrieval_query = expand_legal_query(question_with_context)
        retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        memory = self.store.update(retrieved)
        if self.generator:
            return self.generator.generate(question_with_context, memory)
        return generate_answer(question_with_context, memory)


def answer(question: str, top_k: int) -> str:
    bot = LegalPragChatbot(top_k=top_k, use_llm=False)
    return bot.answer(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-n-adapters", type=int, default=3)
    parser.add_argument("--mode", choices=sorted(GENERATION_MODES), default="hybrid")
    parser.add_argument("--no-llm", action="store_true", help="Use extractive fallback instead of Llama.")
    parser.add_argument("--no-lora", action="store_true", help="Use base Llama without PRAG LoRA adapters.")
    parser.add_argument("--no-dense", action="store_true", help="Disable dense embedding retrieval at runtime.")
    parser.add_argument("--dense-device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    print("Vietnamese Legal Parametric RAG Chatbot")
    mode = "rag" if args.no_lora else args.mode
    if args.no_llm:
        print("Generate mode: extractive fallback")
    elif mode == "rag":
        print(f"Generate mode: base model only ({args.model_name})")
    elif mode == "prag":
        print(f"Generate mode: PRAG LoRA without retrieved source text ({args.model_name})")
    else:
        print(f"Generate mode: hybrid PRAG LoRA + retrieved source text ({args.model_name})")
    print("Gõ 'exit' để thoát.")
    bot = LegalPragChatbot(
        top_k=args.top_k,
        use_llm=not args.no_llm,
        use_lora=not args.no_lora,
        use_dense=not args.no_dense,
        dense_device=args.dense_device,
        mode=mode,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        top_n_adapters=args.top_n_adapters,
    )
    history: list[dict[str, str]] = []
    while True:
        question = input("\nBạn hỏi: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        print()
        answer_text = bot.answer(question, history)
        print(answer_text)
        history.extend(
            [
                {"role": "user", "text": question},
                {"role": "assistant", "text": answer_text},
            ]
        )


if __name__ == "__main__":
    main()
