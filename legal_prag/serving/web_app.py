import argparse
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from legal_prag.serving.chatbot import GENERATION_MODES, LegalPragChatbot
from legal_prag.config import ROOT_DIR
from legal_prag.generation.generator import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_NAME


WEB_DIR = ROOT_DIR / "web"
logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    top_k: int = 8
    mode: str = "hybrid"
    use_llm: bool = True
    use_lora: bool = True
    use_dense: bool = True
    dense_device: str = "cpu"
    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    top_n_adapters: int = 3


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
                        use_dense=self.config.use_dense,
                        dense_device=self.config.dense_device,
                        mode=self.config.mode,
                        model_name=self.config.model_name,
                        max_new_tokens=self.config.max_new_tokens,
                        top_n_adapters=self.config.top_n_adapters,
                    )
        return self._bot

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        with self._lock:
            started = time.perf_counter()
            answer = self.get_bot().answer(question, history)
            logger.info("Answered request in %.2fs", time.perf_counter() - started)
            return answer


def create_app(config: RuntimeConfig) -> FastAPI:
    runtime = ChatRuntime(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.get_bot().warmup()
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
        mode = "extractive" if not config.use_llm else config.mode
        return {
            "mode": mode,
            "top_k": config.top_k,
            "dense_retrieval": config.use_dense,
            "dense_loaded": runtime.get_bot().retriever._embedding_model is not None,
            "dense_device": runtime.get_bot().retriever.dense_device,
            "model_name": config.model_name if config.use_llm else None,
            "max_new_tokens": config.max_new_tokens,
            "top_n_adapters": config.top_n_adapters if config.use_llm and config.use_lora else None,
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
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

    return app


app = create_app(RuntimeConfig())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-n-adapters", type=int, default=5)
    parser.add_argument("--mode", choices=sorted(GENERATION_MODES), default="hybrid")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--dense-device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    mode = "rag" if args.no_lora else args.mode
    config = RuntimeConfig(
        top_k=args.top_k,
        mode=mode,
        use_llm=not args.no_llm,
        use_lora=not args.no_lora,
        use_dense=not args.no_dense,
        dense_device=args.dense_device,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        top_n_adapters=args.top_n_adapters,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
