"""Vietnamese Legal Parametric RAG demo."""

import sys


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


# Backward-compatible module aliases for existing joblib artifacts.
# The source files now live under subpackages, but older pickles may still
# reference the former module paths.
from legal_prag.retrieval import parametric_store as _parametric_store
from legal_prag.retrieval import retriever as _retriever

sys.modules.setdefault("legal_prag.parametric_store", _parametric_store)
sys.modules.setdefault("legal_prag.retriever", _retriever)
