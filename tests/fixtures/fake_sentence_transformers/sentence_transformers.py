"""A stand-in for sentence-transformers, so the examples run in CI without a model download.

It hashes lowercase words into 384 buckets, so texts that share words get a high cosine
similarity. It is for tests only; the examples use the real model.
"""

import re
import zlib

import numpy as np


class SentenceTransformer:
    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, texts, normalize_embeddings: bool = False):
        out = np.zeros((len(texts), 384))
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % 384] += 1.0
        if normalize_embeddings:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
