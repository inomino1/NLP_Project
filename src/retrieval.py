"""
후보 검색(retrieval) 모듈입니다.

구현 목표:
- 1차: TF-IDF(문자 n-gram)로 빠르게 TopN 후보를 축소(lexical)
- 2차: sentence-transformers 임베딩 cosine 유사도로 TopK 정밀 후보 생성(semantic, 옵션)

sentence-transformers가 설치되지 않은 환경에서는 TF-IDF만으로 동작합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .normalize import normalize_text
from .soundbar_db import SoundbarRecord


@dataclass(frozen=True)
class RetrievedCandidate:
    """검색으로 얻은 후보 1개를 표현합니다."""

    canonical: str
    score: float
    lexical_score: float
    semantic_score: Optional[float]


def _safe_import_sklearn() -> tuple[object, object, object]:
    """scikit-learn을 지연 로드합니다(미설치 환경에서 오류 메시지 개선)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import linear_kernel  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "scikit-learn이 필요합니다. `pip install scikit-learn`로 설치해 주세요."
        ) from e
    return TfidfVectorizer, linear_kernel, np


def _safe_import_sentence_transformers():
    """sentence-transformers를 지연 로드합니다(옵션)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    return SentenceTransformer


class HybridRetriever:
    """
    TF-IDF + (옵션) 임베딩 기반 하이브리드 검색기입니다.

    Public API:
    - `fit(records)`: DB 인덱스 구축
    - `retrieve(query)`: TopK 후보 반환
    """

    def __init__(
        self,
        *,
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 1,
        top_n_lexical: int = 200,
        top_k: int = 20,
        embedding_model_id: Optional[str] = "BAAI/bge-small-en-v1.5",
        use_embeddings: bool = True,
        normalize_embeddings: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            char_ngram_range: TF-IDF char n-gram 범위.
            min_df: TF-IDF 최소 문서 빈도.
            top_n_lexical: 1차 lexical 후보 수(2차 임베딩 계산 대상).
            top_k: 최종 반환 후보 수.
            embedding_model_id: sentence-transformers 모델 ID(옵션).
            use_embeddings: True이면 임베딩 기반 2차 검색 시도(미설치면 자동 비활성).
            normalize_embeddings: 임베딩을 L2 정규화하여 cosine을 내적으로 계산.
            random_state: 재현성(일부 모델/연산에서 사용).
        """
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.top_n_lexical = top_n_lexical
        self.top_k = top_k
        self.embedding_model_id = embedding_model_id
        self.use_embeddings = use_embeddings
        self.normalize_embeddings = normalize_embeddings
        self.random_state = random_state

        self._canonicals: list[str] = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._embedder = None
        self._embeddings: Optional[np.ndarray] = None

    @property
    def canonicals(self) -> list[str]:
        """인덱싱된 canonical 모델 문자열 리스트를 반환합니다."""
        return list(self._canonicals)

    def fit(self, records: Iterable[SoundbarRecord]) -> None:
        """
        사운드바 DB 레코드로 인덱스를 구축합니다.

        Args:
            records: SoundbarRecord iterable.

        Raises:
            ValueError: 인덱싱 가능한 canonical이 없을 때.
            ImportError: scikit-learn 미설치 시.
        """
        canonicals = [r.canonical for r in records if r.canonical]
        canonicals = [normalize_text(c) for c in canonicals]
        canonicals = [c for c in canonicals if c]
        if not canonicals:
            raise ValueError("인덱싱할 canonical 모델이 없습니다.")

        TfidfVectorizer, _, _ = _safe_import_sklearn()
        self._canonicals = canonicals
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            lowercase=False,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._canonicals)

        self._maybe_init_embeddings()
        if self._embedder is not None:
            self._embeddings = self._encode_texts(self._canonicals)
        else:
            self._embeddings = None

    def _maybe_init_embeddings(self) -> None:
        """임베딩 모델을 초기화합니다(옵션)."""
        if not self.use_embeddings or not self.embedding_model_id:
            self._embedder = None
            return
        SentenceTransformer = _safe_import_sentence_transformers()
        if SentenceTransformer is None:
            self._embedder = None
            return

        try:
            self._embedder = SentenceTransformer(self.embedding_model_id)
        except Exception:
            # 다운로드/초기화 실패 시에도 TF-IDF만으로 동작 가능해야 함
            self._embedder = None

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩합니다."""
        if self._embedder is None:
            raise RuntimeError("임베딩 모델이 초기화되지 않았습니다.")
        emb = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def retrieve(
        self,
        query: str,
        *,
        top_n_lexical: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedCandidate]:
        """
        주어진 질의 문자열로 후보 TopK를 검색합니다.

        Args:
            query: 질의 문자열(원본 가능, 내부에서 정규화)
            top_n_lexical: 1차 후보 수(override)
            top_k: 최종 후보 수(override)

        Returns:
            RetrievedCandidate 리스트(내림차순 정렬).

        Raises:
            RuntimeError: `fit()`이 호출되지 않았을 때.
        """
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            raise RuntimeError("HybridRetriever.fit()을 먼저 호출해야 합니다.")

        q = normalize_text(query)
        if not q:
            return []

        n_lex = int(top_n_lexical or self.top_n_lexical)
        k = int(top_k or self.top_k)
        n_lex = max(1, min(n_lex, len(self._canonicals)))
        k = max(1, min(k, len(self._canonicals)))

        _, linear_kernel, _ = _safe_import_sklearn()
        q_vec = self._tfidf_vectorizer.transform([q])
        lex_scores = linear_kernel(q_vec, self._tfidf_matrix).ravel()

        # 1차: lexical TopN
        top_lex_idx = np.argpartition(-lex_scores, kth=min(n_lex - 1, len(lex_scores) - 1))[
            :n_lex
        ]
        top_lex_idx = top_lex_idx[np.argsort(-lex_scores[top_lex_idx])]

        # 2차: semantic cosine (옵션)
        sem_scores: Optional[np.ndarray] = None
        if self._embedder is not None and self._embeddings is not None:
            q_emb = self._encode_texts([q])[0]
            cand_emb = self._embeddings[top_lex_idx]
            # normalize_embeddings=True라면 cosine ~= dot
            sem_scores = np.dot(cand_emb, q_emb).astype(np.float32, copy=False)
            rerank_idx = np.argsort(-sem_scores)
            final_idx = top_lex_idx[rerank_idx][:k]
        else:
            final_idx = top_lex_idx[:k]

        out: list[RetrievedCandidate] = []
        for idx in final_idx:
            lex = float(lex_scores[idx])
            sem = None if sem_scores is None else float(sem_scores[np.where(top_lex_idx == idx)[0][0]])
            score = float(sem) if sem is not None else lex
            out.append(
                RetrievedCandidate(
                    canonical=self._canonicals[int(idx)],
                    score=score,
                    lexical_score=lex,
                    semantic_score=sem,
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
            코사인 유사도 [0, 1]. 임베딩 미사용 시 1.0 반환.
        """
        sims = self.compute_similarities_batch(text1, [text2])
        return sims[0] if sims else (1.0 if self._embedder is None else 0.0)

    def compute_similarities_batch(self, query: str, canonicals: list[str]) -> list[float]:
        """
        질의와 여러 canonical 간 코사인 유사도를 1회 인코딩으로 계산합니다.

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
        embs = self._encode_texts(texts)
        q_emb = embs[0]
        return [float(np.dot(q_emb, embs[i + 1])) for i in range(len(canonicals))]

