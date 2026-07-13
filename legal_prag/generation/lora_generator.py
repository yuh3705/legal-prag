from pathlib import Path
import math

from legal_prag.config import ARTIFACT_DIR, LORA_DIR, LORA_MANIFEST_PATH
from legal_prag.generation.generator import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    add_citation,
    build_prompt,
    extract_answer_from_evidence,
    generation_kwargs,
    is_too_vague_answer,
    missing_evidence_answer,
)
from legal_prag.retrieval.parametric_store import TemporaryLegalMemory
from legal_prag.utils import read_json


def build_pure_prag_prompt(question: str) -> str:
    return f"""Tra loi cau hoi phap luat dua tren tri thuc da hoc tu van ban trong adapter LoRA dang duoc kich hoat.
Day la che do PRAG thuan: khong co van ban truy xuat trong prompt, chi dung tham so adapter da duoc chon tu retriever.
Hay tra loi truc tiep, cu the, ngan gon. Neu cau hoi hoi muc phat, neu ro khoang tien, phuong tien/hanh vi ap dung va can cu neu adapter nho duoc.
Khong tra loi chung chung kieu "chua du can cu" khi adapter da co tri thuc lien quan. Chi noi khong ro neu cau hoi hoan toan nam ngoai pham vi phap luat Viet Nam.
Tuyet doi khong dung tieng Trung, tieng Nhat, tieng Han, tieng Anh hoac ngon ngu khac.

Cau hoi: {question}

Tra loi:
"""


class LoraPragGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        include_sources: bool = True,
        top_n_adapters: int = 3,
        adapter_temperature: float = 1.0,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.PeftModel = PeftModel
        self.max_new_tokens = max_new_tokens
        self.include_sources = include_sources
        self.top_n_adapters = max(1, top_n_adapters)
        self.adapter_temperature = max(0.001, adapter_temperature)
        self.manifest = read_json(LORA_MANIFEST_PATH) if Path(LORA_MANIFEST_PATH).exists() else {"adapters": {}}
        self.adapters = self.manifest.get("adapters", {})

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.base_model.to("cpu")
        self.model = None
        self.loaded_adapters: set[str] = set()
        self.active_fused_adapter: str | None = None

    def resolve_adapter_path(self, chunk_id: str, adapter_path: str) -> str:
        path = Path(adapter_path)
        if path.exists():
            return str(path)

        adapter_dir = chunk_id.replace("::", "_")
        model_dir = DEFAULT_MODEL_NAME.replace("/", "_")
        candidates = [
            LORA_DIR / model_dir / adapter_dir,
            ARTIFACT_DIR / "lora_adapters" / model_dir / adapter_dir,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            f"LoRA adapter for chunk {chunk_id} not found. Manifest path was {adapter_path}."
        )

    def pick_adapters(self, memory: TemporaryLegalMemory) -> list[tuple[str, str, float]]:
        selected: list[tuple[str, str, float]] = []
        seen_chunk_ids: set[str] = set()
        for item in memory.evidence:
            chunk_id = item.chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            adapter = self.adapters.get(chunk_id)
            if adapter:
                selected.append((chunk_id, adapter["path"], float(item.score)))
                seen_chunk_ids.add(chunk_id)
            if len(selected) >= self.top_n_adapters:
                break
        return selected

    def adapter_name(self, chunk_id: str) -> str:
        return chunk_id.replace(":", "_").replace("/", "_").replace("\\", "_")

    def adapter_weights(self, adapters: list[tuple[str, str, float]]) -> list[float]:
        if not adapters:
            return []
        scores = [score / self.adapter_temperature for _, _, score in adapters]
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(adapters)] * len(adapters)
        return [value / total for value in exps]

    def ensure_adapter_loaded(self, chunk_id: str, adapter_path: str) -> str:
        adapter_name = self.adapter_name(chunk_id)
        if self.model is None:
            self.model = self.PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name,
            )
            self.loaded_adapters.add(adapter_name)
        elif adapter_name not in self.loaded_adapters:
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
            self.loaded_adapters.add(adapter_name)
        return adapter_name

    def weighted_adapter_host(self):
        if self.model is None:
            return None
        for candidate in (self.model, getattr(self.model, "base_model", None)):
            if candidate is not None and hasattr(candidate, "add_weighted_adapter"):
                return candidate
        return None

    def delete_active_fused_adapter(self) -> None:
        if self.model is None or not self.active_fused_adapter:
            return
        adapter_name = self.active_fused_adapter
        for candidate in (self.model, getattr(self.model, "base_model", None)):
            if candidate is not None and hasattr(candidate, "delete_adapter"):
                try:
                    candidate.delete_adapter(adapter_name)
                    self.loaded_adapters.discard(adapter_name)
                    self.active_fused_adapter = None
                    return
                except Exception:
                    continue

    def activate_adapters(self, adapters: list[tuple[str, str, float]]) -> None:
        loaded_names = []
        resolved_adapters = []
        for chunk_id, adapter_path, score in adapters:
            resolved_path = self.resolve_adapter_path(chunk_id, adapter_path)
            loaded_names.append(self.ensure_adapter_loaded(chunk_id, resolved_path))
            resolved_adapters.append((chunk_id, resolved_path, score))

        if self.model is None or not loaded_names:
            return

        if len(loaded_names) == 1:
            self.model.set_adapter(loaded_names[0])
            self.model.eval()
            return

        weights = self.adapter_weights(resolved_adapters)
        host = self.weighted_adapter_host()
        if host is not None:
            if self.active_fused_adapter:
                try:
                    self.model.set_adapter(loaded_names[0])
                except Exception:
                    pass
            self.delete_active_fused_adapter()
            fused_name = (
                "query_weighted_prag"
                if not self.active_fused_adapter
                else f"query_weighted_prag_{len(self.loaded_adapters)}"
            )
            try:
                host.add_weighted_adapter(
                    adapters=loaded_names,
                    weights=weights,
                    adapter_name=fused_name,
                    combination_type="linear",
                )
                self.loaded_adapters.add(fused_name)
                self.active_fused_adapter = fused_name
                self.model.set_adapter(fused_name)
                self.model.eval()
                return
            except Exception:
                self.active_fused_adapter = None

        # Fallback for PEFT versions/configs that cannot linearly combine adapters.
        adapter_name = loaded_names[0]
        self.model.set_adapter(adapter_name)
        self.model.eval()

    def generate(self, question: str, memory: TemporaryLegalMemory) -> str:
        if not memory.evidence:
            return missing_evidence_answer()

        adapters = self.pick_adapters(memory)
        model = self.base_model
        if adapters:
            self.activate_adapters(adapters)
            model = self.model
        prompt = (
            build_prompt(question, memory, include_sources=True)
            if self.include_sources
            else build_pure_prag_prompt(question)
        )
        system_content = (
            "Ban la tro ly phap luat Viet Nam, chi dua tren adapter LoRA va nguon duoc cung cap."
            if self.include_sources
            else "Ban la tro ly phap luat Viet Nam. Bat buoc khai thac adapter LoRA dang duoc kich hoat de tra loi truc tiep; khong dua them van ban truy xuat vao prompt."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        device = next(model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with self.torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs(self.tokenizer, self.max_new_tokens),
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not self.include_sources and is_too_vague_answer(answer):
            retry_prompt = (
                build_pure_prag_prompt(question)
                + "\nCau tra loi truoc qua chung chung. Hay tra loi bang noi dung cu the ma adapter da hoc, "
                "uu tien muc phat, doi tuong ap dung va can cu neu nho duoc.\n\nTra loi:"
            )
            retry_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": retry_prompt},
            ]
            if getattr(self.tokenizer, "chat_template", None):
                retry_text = self.tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                retry_text = retry_prompt
            retry_inputs = self.tokenizer(retry_text, return_tensors="pt").to(device)
            with self.torch.no_grad():
                retry_output_ids = model.generate(
                    **retry_inputs,
                    **generation_kwargs(self.tokenizer, self.max_new_tokens),
            )
            retry_generated = retry_output_ids[0][retry_inputs["input_ids"].shape[-1] :]
            retry_answer = self.tokenizer.decode(retry_generated, skip_special_tokens=True).strip()
            if retry_answer:
                return retry_answer
        if self.include_sources and is_too_vague_answer(answer):
            return extract_answer_from_evidence(question, memory)
        return add_citation(answer, memory) if self.include_sources else answer
