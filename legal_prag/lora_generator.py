from pathlib import Path

from legal_prag.config import LORA_MANIFEST_PATH
from legal_prag.generator import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    add_citation,
    build_llama_prompt,
    generation_kwargs,
    missing_evidence_answer,
)
from legal_prag.parametric_store import TemporaryLegalMemory
from legal_prag.utils import read_json


class LoraPragGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.PeftModel = PeftModel
        self.max_new_tokens = max_new_tokens
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

    def pick_adapter(self, memory: TemporaryLegalMemory) -> tuple[str | None, str | None]:
        for item in memory.evidence:
            chunk_id = item.chunk["chunk_id"]
            adapter = self.adapters.get(chunk_id)
            if adapter:
                return chunk_id, adapter["path"]
        return None, None

    def activate_adapter(self, chunk_id: str, adapter_path: str) -> None:
        adapter_name = chunk_id.replace(":", "_").replace("/", "_")
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
        self.model.set_adapter(adapter_name)
        self.model.eval()

    def generate(self, question: str, memory: TemporaryLegalMemory) -> str:
        if not memory.evidence:
            return missing_evidence_answer()

        chunk_id, adapter_path = self.pick_adapter(memory)
        model = self.base_model
        if adapter_path is not None:
            self.activate_adapter(chunk_id, adapter_path)
            model = self.model
        prompt = build_llama_prompt(question, memory)
        messages = [
            {"role": "system", "content": "Ban la tro ly phap luat Viet Nam, chi dua tren adapter LoRA va nguon duoc cung cap."},
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
        return add_citation(answer, memory)
