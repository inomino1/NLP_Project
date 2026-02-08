"""
정확도 우선 임베딩 기반 검색 모듈입니다.

- TF-IDF 프리필터 없이 전체 DB에 대해 임베딩 코사인 유사도로 랭킹
- 브랜드 필터(옵션): 추출된 brand로 후보 풀 축소
- 속도보다 정확도에 중점
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .normalize import normalize_text
from .soundbar_db import SoundbarRecord


@dataclass(frozen=True)
class EmbeddingCandidate:
    """임베딩 검색 후보 1개를 표현합니다."""

    canonical: str
    score: float
    brand: str
    model_part: str


def _safe_import_sentence_transformers():
    """sentence-transformers를 지연 로드합니다."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    return SentenceTransformer


class EmbeddingRetriever:
    """
    임베딩 + 코사인 유사도 기반 검색기입니다.
    정확도 우선으로 TF-IDF 프리필터 없이 전체 DB에 대해 검색합니다.
    """

    def __init__(
        self,
        *,
        embedding_model_id: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        top_k: int = 20,
    ) -> None:
        """
        Args:
            embedding_model_id: sentence-transformers 모델 ID.
            normalize_embeddings: L2 정규화하여 cosine을 내적으로 계산.
            top_k: 반환 후보 수.
        """
        self.embedding_model_id = embedding_model_id
        self.normalize_embeddings = normalize_embeddings
        self.top_k = top_k

        self._embedder = None
        self._records: list[SoundbarRecord] = []
        self._canonicals: list[str] = []
        self._brands: list[str] = []
        self._model_parts: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, records: Iterable[SoundbarRecord]) -> None:
        """
        사운드바 DB 레코드로 인덱스를 구축합니다.

        Args:
            records: SoundbarRecord iterable.
        """
        self._records = list(records)
        self._canonicals = [r.canonical for r in self._records if r.canonical]
        self._brands = [r.brand or "" for r in self._records if r.canonical]
        self._model_parts = [
            (r.model or r.canonical or "").strip() for r in self._records if r.canonical
        ]

        if not self._canonicals:
            raise ValueError("인덱싱할 canonical 모델이 없습니다.")

        SentenceTransformer = _safe_import_sentence_transformers()
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers가 필요합니다. `pip install sentence-transformers`로 설치해 주세요."
            )

        self._embedder = SentenceTransformer(self.embedding_model_id)

        # 모델명 검색 정확도 향상: DB의 model_part만 임베딩 (full canonical 대신)
        texts_to_embed = [normalize_text(mp) for mp in self._model_parts]
        self._embeddings = self._embedder.encode(
            texts_to_embed,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        ).astype(np.float32)

    def retrieve(
        self,
        query: str,
        *,
        brand_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[EmbeddingCandidate]:
        """
        질의에 대해 임베딩 코사인 유사도로 TopK 후보를 반환합니다.

        Args:
            query: 질의 문자열(기기명 또는 모델 부분).
            brand_filter: 지정 시 해당 브랜드 후보만 검색.
            top_k: 반환 후보 수(override).

        Returns:
            EmbeddingCandidate 리스트(내림차순).
        """
        if self._embedder is None or self._embeddings is None:
            raise RuntimeError("EmbeddingRetriever.fit()을 먼저 호출해야 합니다.")

        q = normalize_text(query)
        if not q:
            return []

        k = int(top_k or self.top_k)
        k = max(1, min(k, len(self._canonicals)))

        # 브랜드 필터 적용
        if brand_filter:
            brand_norm = normalize_text(brand_filter)
            indices = [i for i in range(len(self._canonicals)) if self._brands[i] == brand_norm]
            if not indices:
                indices = list(range(len(self._canonicals)))
        else:
            indices = list(range(len(self._canonicals)))

        # 질의 임베딩
        q_emb = self._embedder.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        # 코사인 유사도 (정규화된 내적)
        cand_emb = self._embeddings[indices]
        scores = np.dot(cand_emb, q_emb).astype(np.float64)

        # TopK
        top_idx = np.argsort(-scores)[:k]
        out: list[EmbeddingCandidate] = []
        for i in top_idx:
            idx = indices[i]
            out.append(
                EmbeddingCandidate(
                    canonical=self._canonicals[idx],
                    score=float(scores[i]),
                    brand=self._brands[idx],
                    model_part=self._model_parts[idx],
                )
            )
        return out

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        두 문자열 간 코사인 유사도를 계산합니다.

        primary_query와 top1 candidate 간 직접 유사도 검증에 사용합니다.

        Args:
            text1: 첫 번째 문자열(예: primary_query)
            text2: 두 번째 문자열(예: canonical 모델명)

        Returns:
            코사인 유사도 [0, 1].
        """
        sims = self.compute_similarities_batch(text1, [text2])
        return sims[0] if sims else 0.0

    def compute_similarities_batch(self, query: str, canonicals: list[str]) -> list[float]:
        """
        질의와 여러 canonical 간 코사인 유사도를 1회 인코딩으로 계산합니다.
        (속도 개선: N회 별도 encode 대신 1회 배치 encode)

        Args:
            query: 질의 문자열
            canonicals: canonical 모델명 리스트

        Returns:
            각 canonical에 대한 유사도 리스트
        """
        if self._embedder is None or not canonicals:
            return [1.0] * len(canonicals) if canonicals else []
        q = normalize_text(query) or query
        if not q:
            return [0.0] * len(canonicals)
        texts = [q] + [normalize_text(c) or c for c in canonicals]
        embs = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)
        q_emb = embs[0]
        return [float(np.dot(q_emb, embs[i + 1])) for i in range(len(canonicals))]
