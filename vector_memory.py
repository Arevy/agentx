from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class MemoryHit:
    text: str
    metadata: dict
    score: float


class VectorMemory:
    """
    Lightweight FAISS-backed vector store for agent memory and notes.

    Stores embeddings and metadata on disk so the agent can retrieve relevant
    context at each turn.
    """

    def __init__(
        self,
        store_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "metadata.jsonl"
        self.embedding_model_name = embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        self.metadata: List[dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._dimension: Optional[int] = None
        self._load()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vector = self._embed(["bootstrap"])
            self._dimension = vector.shape[1]
        return self._dimension

    def _ensure_index(self) -> faiss.IndexFlatIP:
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        return self.index

    def _embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype("float32")

    def _load(self) -> None:
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self.metadata.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if self.index_path.exists() and self.metadata:
            self.index = faiss.read_index(str(self.index_path))
            self._dimension = self.index.d
        else:
            # Rebuild from metadata if index missing.
            if self.metadata:
                texts = [item["text"] for item in self.metadata]
                vectors = self._embed(texts)
                self.index = faiss.IndexFlatIP(vectors.shape[1])
                self.index.add(vectors)
                faiss.write_index(self.index, str(self.index_path))

    def add(self, text: str, metadata: Optional[dict] = None) -> None:
        if not text.strip():
            return
        metadata = metadata or {}
        record = {"text": text, "metadata": metadata}
        vector = self._embed([text])
        index = self._ensure_index()
        index.add(vector)
        self.metadata.append(record)
        with self.meta_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        faiss.write_index(index, str(self.index_path))

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not query.strip() or self.index is None or not self.metadata:
            return []
        vector = self._embed([query])
        top_k = max(1, min(top_k, len(self.metadata)))
        scores, indices = self.index.search(vector, top_k)
        hits: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            hits.append(
                {
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "score": float(score),
                }
            )
        return hits

