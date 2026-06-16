import json
import re
import unicodedata
from html import unescape
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(html: str) -> str:
    html = html or ""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n", html)
    html = re.sub(r"(?i)</div\s*>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return normalize_text(unescape(text))


def read_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_nested(data: dict, dotted_path: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def pick_first(data: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        value = get_nested(data, key, None)
        if value:
            return str(value)
    return default
