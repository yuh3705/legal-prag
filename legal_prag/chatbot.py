import argparse
from typing import Any

from legal_prag.config import PARAMETRIC_STORE_PATH, RETRIEVER_PATH
from legal_prag.generator import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_NAME, LlamaGenerator, generate_answer
from legal_prag.lora_generator import LoraPragGenerator
from legal_prag.parametric_store import ParametricStore
from legal_prag.retriever import LegalRetriever


def format_history(history: list[dict[str, Any]] | None, max_turns: int = 3, max_chars: int = 1200) -> str:
    if not history:
        return ""

    lines = []
    for item in history[-max_turns * 2 :]:
        role = str(item.get("role", "")).strip().lower()
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        label = "Nguoi dung" if role == "user" else "Tro ly"
        lines.append(f"{label}: {text}")

    formatted = "\n".join(lines)
    if len(formatted) <= max_chars:
        return formatted
    return formatted[-max_chars:]


def contextual_question(question: str, history: list[dict[str, Any]] | None = None) -> str:
    context = format_history(history)
    if not context:
        return question
    return f"""Ngu canh hoi dap truoc:
{context}

Cau hoi hien tai:
{question}"""


class LegalPragChatbot:
    def __init__(
        self,
        top_k: int = 5,
        use_llm: bool = True,
        use_lora: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.top_k = top_k
        self.retriever = LegalRetriever.load(str(RETRIEVER_PATH))
        self.store = ParametricStore.load(str(PARAMETRIC_STORE_PATH))
        if not use_llm:
            self.generator = None
        elif use_lora:
            self.generator = LoraPragGenerator(model_name, max_new_tokens)
        else:
            self.generator = LlamaGenerator(model_name, max_new_tokens)

    def answer(self, question: str, history: list[dict[str, Any]] | None = None) -> str:
        question_with_context = contextual_question(question, history)
        retrieved = self.retriever.search(question_with_context, top_k=self.top_k)
        memory = self.store.update(retrieved)
        if self.generator:
            return self.generator.generate(question_with_context, memory)
        return generate_answer(question, memory)


def answer(question: str, top_k: int) -> str:
    bot = LegalPragChatbot(top_k=top_k, use_llm=False)
    return bot.answer(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--no-llm", action="store_true", help="Use extractive fallback instead of Llama.")
    parser.add_argument("--no-lora", action="store_true", help="Use base Llama without PRAG LoRA adapters.")
    args = parser.parse_args()

    print("Vietnamese Legal Parametric RAG Chatbot")
    if args.no_llm:
        print("Generate mode: extractive fallback")
    elif args.no_lora:
        print(f"Generate mode: base model only ({args.model_name})")
    else:
        print(f"Generate mode: PRAG LoRA adapters + {args.model_name}")
    print("Gõ 'exit' để thoát.")
    bot = LegalPragChatbot(
        top_k=args.top_k,
        use_llm=not args.no_llm,
        use_lora=not args.no_lora,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
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
