import argparse
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from legal_prag.chatbot import LegalPragChatbot
from legal_prag.config import ROOT_DIR
from legal_prag.generator import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_NAME


WEB_DIR = ROOT_DIR / "web"


@dataclass
class RuntimeConfig:
    top_k: int = 5
    use_llm: bool = True
    use_lora: bool = True
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    answer: str


class ChatRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._bot: LegalPragChatbot | None = None
        self._lock = threading.RLock()

    def get_bot(self) -> LegalPragChatbot:
        if self._bot is None:
            with self._lock:
                if self._bot is None:
                    self._bot = LegalPragChatbot(
                        top_k=self.config.top_k,
                        use_llm=self.config.use_llm,
                        use_lora=self.config.use_lora,
                        model_name=self.config.model_name,
                        max_new_tokens=self.config.max_new_tokens,
                    )
        return self._bot

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        with self._lock:
            return self.get_bot().answer(question, history)


def create_app(config: RuntimeConfig) -> FastAPI:
    runtime = ChatRuntime(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.get_bot()
        yield

    app = FastAPI(title="Vietnamese Legal PRAG Chatbot", lifespan=lifespan)
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status():
        mode = "extractive"
        if config.use_llm and config.use_lora:
            mode = "prag_lora"
        elif config.use_llm:
            mode = "base_llama"
        return {
            "mode": mode,
            "top_k": config.top_k,
            "model_name": config.model_name if config.use_llm else None,
            "max_new_tokens": config.max_new_tokens,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")
        try:
            return ChatResponse(answer=runtime.answer(question, payload.history))
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Missing artifact: {exc}. Run prepare_corpus and build_store first.",
            ) from exc

    return app


app = create_app(RuntimeConfig())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig(
        top_k=args.top_k,
        use_llm=not args.no_llm,
        use_lora=not args.no_lora,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
