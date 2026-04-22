"""Lightweight RAG retriever for the AutoStream knowledge base.

Uses TF-IDF-style keyword scoring over a local JSON knowledge base. This keeps
the project dependency-light (no vector DB, no embedding API calls) while still
demonstrating a real retrieval step before generation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Dict

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "and",
    "or", "in", "on", "at", "with", "by", "it", "this", "that", "do", "does",
    "i", "you", "me", "my", "your", "we", "our", "be", "have", "has", "can",
    "tell", "about", "what", "how", "much", "plan", "plans",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class KnowledgeBase:
    """In-memory BM25-lite retriever backed by knowledge_base.json."""

    def __init__(self, path: str | Path = "knowledge_base.json") -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.docs: List[Dict] = data["documents"]
        self._doc_tokens: List[List[str]] = [
            _tokenize(f"{d['topic']} {d['content']}") for d in self.docs
        ]
        self._df: Dict[str, int] = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self.docs)

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0
        score = 0.0
        doc_len = len(doc_tokens)
        for term in query_tokens:
            tf = doc_tokens.count(term)
            if tf == 0:
                continue
            idf = math.log((self._n + 1) / (self._df.get(term, 0) + 1)) + 1
            score += idf * (tf / doc_len)
        return score

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        ranked = sorted(
            (
                (self._score(query_tokens, tokens), doc)
                for tokens, doc in zip(self._doc_tokens, self.docs)
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for score, doc in ranked[:k] if score > 0]

    def format_context(self, docs: List[Dict]) -> str:
        if not docs:
            return "(no relevant knowledge base entries found)"
        return "\n\n".join(f"[{d['topic']}]\n{d['content']}" for d in docs)
