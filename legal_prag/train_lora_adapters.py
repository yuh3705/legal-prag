import argparse
import gc
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from legal_prag.config import CHUNKS_JSON, LORA_DIR, LORA_MANIFEST_PATH, ensure_dirs
from legal_prag.generator import DEFAULT_MODEL_NAME
from legal_prag.utils import read_json, write_json


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value[:120].strip("_") or "adapter"


def model_lora_dir(model_name: str) -> Path:
    return LORA_DIR / safe_name(model_name)


def target_modules(args) -> list[str]:
    return [item.strip() for item in args.target_modules.split(",") if item.strip()]


def compact_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", compact_text(text, max_chars=20000))
    return [part.strip() for part in parts if len(part.strip()) >= 40]


def metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def augment_chunk(chunk: dict, max_chars: int) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    title = chunk.get("title") or chunk["doc_id"]
    so_ky_hieu = metadata_value(metadata, "so_ky_hieu", "so_hieu")
    loai_van_ban = metadata_value(metadata, "loai_van_ban")
    co_quan = metadata_value(metadata, "co_quan_ban_hanh")
    ngay_ban_hanh = metadata_value(metadata, "ngay_ban_hanh")
    tinh_trang = metadata_value(metadata, "tinh_trang_hieu_luc")
    text = compact_text(chunk["text"], max_chars=max_chars)
    sentences = sentence_split(text)

    source_lines = [
        f"Ten van ban: {title}",
        f"Ma chunk: {chunk['chunk_id']}",
    ]
    if so_ky_hieu:
        source_lines.append(f"So ky hieu: {so_ky_hieu}")
    if loai_van_ban:
        source_lines.append(f"Loai van ban: {loai_van_ban}")
    if co_quan:
        source_lines.append(f"Co quan ban hanh: {co_quan}")
    if ngay_ban_hanh:
        source_lines.append(f"Ngay ban hanh: {ngay_ban_hanh}")
    if tinh_trang:
        source_lines.append(f"Tinh trang hieu luc: {tinh_trang}")

    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": title,
        "metadata": {
            "so_ky_hieu": so_ky_hieu,
            "loai_van_ban": loai_van_ban,
            "co_quan_ban_hanh": co_quan,
            "ngay_ban_hanh": ngay_ban_hanh,
            "tinh_trang_hieu_luc": tinh_trang,
        },
        "source_card": "\n".join(source_lines),
        "passage": text,
        "sentences": sentences,
    }


def answer_with_source(answer: str, augmented: dict[str, Any]) -> str:
    source = augmented["source_card"]
    return f"{compact_text(answer, max_chars=1800)}\n\nCăn cứ:\n{source}"


def make_qa_examples(augmented: dict[str, Any], qa_per_chunk: int) -> list[dict[str, str]]:
    title = augmented["title"]
    metadata = augmented["metadata"]
    passage = augmented["passage"]
    sentences = augmented["sentences"]
    source_card = augmented["source_card"]
    so_ky_hieu = metadata.get("so_ky_hieu") or "khong ro"
    co_quan = metadata.get("co_quan_ban_hanh") or "khong ro"
    ngay_ban_hanh = metadata.get("ngay_ban_hanh") or "khong ro"

    candidates = [
        {
            "instruction": f"Van ban {title} quy dinh noi dung gi?",
            "answer": answer_with_source(passage, augmented),
        },
        {
            "instruction": f"Tom tat can cu phap ly cua van ban {title}.",
            "answer": answer_with_source(passage, augmented),
        },
        {
            "instruction": f"So ky hieu, co quan ban hanh va ngay ban hanh cua van ban {title} la gi?",
            "answer": answer_with_source(
                f"So ky hieu: {so_ky_hieu}. Co quan ban hanh: {co_quan}. Ngay ban hanh: {ngay_ban_hanh}.",
                augmented,
            ),
        },
        {
            "instruction": "Hay tra loi cau hoi phap luat dua tren tri thuc da hoc tu van ban va neu can cu.",
            "answer": answer_with_source(passage, augmented),
        },
        {
            "instruction": f"Nhung diem chinh can nho trong chunk cua van ban {title} la gi?",
            "answer": answer_with_source("\n".join(f"- {item}" for item in sentences[:5]) or passage, augmented),
        },
        {
            "instruction": f"Khi duoc hoi ve {title}, can dua ra can cu nao?",
            "answer": source_card,
        },
    ]

    for idx, sentence in enumerate(sentences[: max(0, qa_per_chunk - len(candidates))], start=1):
        candidates.append(
            {
                "instruction": f"Can cu thu {idx} trong van ban {title} noi gi?",
                "answer": answer_with_source(sentence, augmented),
            }
        )

    return candidates[:qa_per_chunk]


def build_training_examples(chunk: dict, args) -> list[dict[str, str]]:
    augmented = augment_chunk(chunk, args.max_text_chars)
    return make_qa_examples(augmented, args.qa_per_chunk)


def write_training_examples(adapter_dir: Path, chunk: dict, examples: list[dict[str, str]]) -> None:
    payload = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk.get("title", ""),
        "examples": examples,
    }
    write_json(adapter_dir / "training_qa.json", payload)


def make_examples(chunk: dict, max_chars: int) -> list[dict[str, str]]:
    augmented = augment_chunk(chunk, max_chars)
    return [
        {
            "instruction": f"Van ban {augmented['title']} quy dinh noi dung gi?",
            "answer": answer_with_source(augmented["passage"], augmented),
        },
        {
            "instruction": f"Tom tat can cu phap ly trong van ban {augmented['title']}.",
            "answer": answer_with_source(augmented["passage"], augmented),
        },
        {
            "instruction": "Tra loi cau hoi phap luat dua tren tri thuc da hoc tu van ban.",
            "answer": answer_with_source(augmented["passage"], augmented),
        },
    ]


def format_chat(tokenizer, instruction: str, answer: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Ban la tro ly phap luat Viet Nam. Tra loi ngan gon va dua tren van ban da hoc.",
        },
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"System: {messages[0]['content']}\nUser: {instruction}\nAssistant: {answer}"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.items = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = item["input_ids"].clone()
            item["labels"][item["attention_mask"] == 0] = -100
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def build_lora_config(args) -> LoraConfig:
    return LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules(args),
    )


def ensure_init_adapter(args, base_model, init_adapter_dir: Path):
    adapter_file = init_adapter_dir / "adapter_model.safetensors"
    if adapter_file.exists():
        return base_model

    print(f"No LoRA base weight, creating: {init_adapter_dir}")
    model = get_peft_model(base_model, build_lora_config(args))
    model.save_pretrained(init_adapter_dir)
    base_model = model.unload()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not adapter_file.exists():
        raise RuntimeError(f"Failed to create LoRA base weight: {adapter_file}")
    return base_model


def train_one_adapter(args, chunk: dict, model, tokenizer, adapter_dir: Path) -> None:
    examples = build_training_examples(chunk, args)
    write_training_examples(adapter_dir, chunk, examples)
    texts = [format_chat(tokenizer, item["instruction"], item["answer"]) for item in examples]
    dataset = TextDataset(texts, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "trainer_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available() and not args.force_cpu and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and not args.force_cpu and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del trainer, dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunks", default=str(CHUNKS_JSON))
    parser.add_argument("--max-adapters", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-text-chars", type=int, default=2400)
    parser.add_argument("--qa-per-chunk", type=int, default=8)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    lora_dir = model_lora_dir(args.model_name)
    lora_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = read_json(args.chunks)
    chunks = all_chunks if args.max_adapters <= 0 else all_chunks[: args.max_adapters]
    if torch.cuda.is_available() and not args.force_cpu:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    if args.force_cpu or not torch.cuda.is_available():
        base_model.to("cpu")
    base_model.config.use_cache = False

    init_adapter_dir = lora_dir / f"base_weight_rank={args.rank}_alpha={args.alpha}"
    base_model = ensure_init_adapter(args, base_model, init_adapter_dir)

    manifest = {
        "base_model": args.model_name,
        "note": "One LoRA adapter is trained per retrieved chunk/document, PRAG-style.",
        "init_adapter": str(init_adapter_dir),
        "adapters": {},
    }

    for chunk in tqdm(chunks, desc="Training LoRA adapters"):
        adapter_name = safe_name(chunk["chunk_id"])
        adapter_dir = lora_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        if (adapter_dir / "adapter_model.safetensors").exists() and not args.force_retrain:
            manifest["adapters"][chunk["chunk_id"]] = {
                "path": str(adapter_dir),
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
            }
            write_json(LORA_MANIFEST_PATH, manifest)
            continue

        model = PeftModel.from_pretrained(base_model, init_adapter_dir, is_trainable=True)
        train_one_adapter(args, chunk, model, tokenizer, adapter_dir)
        base_model = model.unload()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manifest["adapters"][chunk["chunk_id"]] = {
            "path": str(adapter_dir),
            "doc_id": chunk["doc_id"],
            "title": chunk.get("title", ""),
        }
        write_json(LORA_MANIFEST_PATH, manifest)

    print(f"Saved LoRA manifest: {LORA_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
