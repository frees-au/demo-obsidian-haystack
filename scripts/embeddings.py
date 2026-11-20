"""Deterministic word-hashing vectors; no model or network access required."""

import hashlib
import math
import re
from collections import Counter
from dataclasses import replace

from haystack import Document, component

EMBEDDING_DIM = 384
INDEX_NAME = "haystack_word_hash_v1"


def embed(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for word, count in Counter(re.findall(r"\w+", text.casefold())).items():
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:7], "big") % EMBEDDING_DIM
        sign = 1 if digest[7] & 1 else -1
        vector[bucket] += sign * count
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


@component
class LocalTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str):
        return {"embedding": embed(text)}


@component
class LocalDocumentEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": [
            replace(document, embedding=embed(document.content or ""))
            for document in documents
        ]}
