"""Shared sentence-transformer and collection settings."""

from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
INDEX_NAME = "haystack_minilm_l6_v2"
STORE_PATH = Path(__file__).resolve().parents[1] / ".haystack" / "qdrant"
