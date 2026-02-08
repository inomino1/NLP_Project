"""
사운드바 모델 매핑 Agent(오케스트레이션) 모듈입니다.

파이프라인:
1) 사운드바 DB 로드/정규화
2) 로그 행에서 후보 질의 생성
3) 하이브리드 검색(TF-IDF + 임베딩)으로 후보 TopK 생성
4) verifier로 UNKNOWN/모델 결정
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .brand_extractor import BrandExtraction, extract_brand_from_query
from .embedding_retriever import EmbeddingRetriever
from .log_features import (
    LogQuery,
    TypedQuery,
    build_log_queries_from_row,
    build_typed_queries_from_row,
    choose_primary_query,
)
from .retrieval import HybridRetriever, RetrievedCandidate
from .soundbar_db import SoundbarRecord, get_brand_set, load_soundbar_db_from_py
from .verify import (
    Prediction,
    RuleBasedVerifier,
    VerificationConfig,
    _get_canonical_model_part,
)


@dataclass(frozen=True)
class AgentConfig:
    """Agent 구성/하이퍼파라미터 설정입니다."""

    top_n_lexical: int = 200
    top_k: int = 20
    embedding_model_id: Optional[str] = "BAAI/bge-small-en-v1.5"
    use_embeddings: bool = True
    include_bt: bool = True
    accept_score_threshold: float = 0.52  # 0.55→0.52 (LG NS60TR 등 BT 가중치 완화)
    margin_threshold: float = 0.05
    accuracy_mode: bool = False  # True: NER+임베딩 전용, 정확도 우선
    min_cosine_similarity: Optional[float] = 0.85  # primary_query vs top1 직접 유사도. None이면 비활성
    cosine_relaxed_with_overlap: float = 0.70  # overlap 있을 때 fallback 검사용 완화 임계치 (0.65→0.70, 오탐 감소)
    cosine_gate_max_candidates: int = 5  # cosine gate에서 검사하는 최대 후보 수
    min_lexical_similarity: Optional[float] = 0.3  # 모델 파트 렉시컬 유사도 미만이면 UNKNOWN. None이면 비활성


@dataclass(frozen=True)
class PredictionResult:
    """
    Agent의 단일 row 예측 결과입니다.

    Attributes:
        row_id: 원본 데이터 행 식별자(없으면 None)
        predicted: 예측 canonical 모델명(UNKNOWN이면 None)
        confidence: 신뢰도
        primary_query: 대표 질의(정규화 문자열)
        candidates: 상위 후보 리스트(내림차순)
        evidence: verifier에서 반환한 근거 dict
    """

    row_id: Optional[int]
    predicted: Optional[str]
    confidence: float
    primary_query: str
    candidates: list[RetrievedCandidate]
    evidence: dict[str, Any]


class SoundbarModelAgent:
    """
    사운드바 모델 매핑 Agent 입니다.

    Public API:
    - `predict_row(row_dict, row_id=None) -> PredictionResult`
    - `batch_predict(df) -> pd.DataFrame`
    """

    def __init__(self, soundbar_list_py: Path, config: Optional[AgentConfig] = None) -> None:
        """
        Args:
            soundbar_list_py: `soundbar_list.py` 경로
            config: AgentConfig
        """
        self.soundbar_list_py = soundbar_list_py
        self.config = config or AgentConfig()

        self._records: list[SoundbarRecord] = []
        self._brand_set: set[str] = set()
        if self.config.accuracy_mode:
            self._embedding_retriever = EmbeddingRetriever(
                embedding_model_id=self.config.embedding_model_id or "BAAI/bge-small-en-v1.5",
                top_k=self.config.top_k,
            )
            self._hybrid_retriever: Optional[HybridRetriever] = None
        else:
            self._embedding_retriever = None
            self._hybrid_retriever = HybridRetriever(
                top_n_lexical=self.config.top_n_lexical,
                top_k=self.config.top_k,
                embedding_model_id=self.config.embedding_model_id,
                use_embeddings=self.config.use_embeddings,
            )
        self._verifier: Optional[RuleBasedVerifier] = None

        self._load_and_build()

    def _load_and_build(self) -> None:
        """사운드바 DB 로드 및 인덱스를 구축합니다."""
        self._records = load_soundbar_db_from_py(self.soundbar_list_py)
        self._brand_set = get_brand_set(self._records)
        if self._embedding_retriever is not None:
            self._embedding_retriever.fit(self._records)
        elif self._hybrid_retriever is not None:
            self._hybrid_retriever.fit(self._records)

        self._verifier = RuleBasedVerifier(
            config=VerificationConfig(
                accept_score_threshold=self.config.accept_score_threshold,
                margin_threshold=self.config.margin_threshold,
                known_brands=self._brand_set,
            )
        )

    @property
    def brand_set(self) -> set[str]:
        """DB로부터 추출된 브랜드 집합을 반환합니다."""
        return set(self._brand_set)

    def _embedding_cands_to_retrieved(
        self, cands: list, weights: float = 1.0
    ) -> list[RetrievedCandidate]:
        """EmbeddingCandidate를 RetrievedCandidate로 변환합니다."""
        out: list[RetrievedCandidate] = []
        for c in cands:
            wscore = float(c.score) * float(weights)
            out.append(
                RetrievedCandidate(
                    canonical=c.canonical,
                    score=wscore,
                    lexical_score=0.0,
                    semantic_score=c.score,
                )
            )
        return out

    def _retrieve_accuracy(self, queries: list[LogQuery]) -> list[RetrievedCandidate]:
        """정확도 모드: NER로 브랜드 추출 후 임베딩 검색."""
        best: dict[str, RetrievedCandidate] = {}
        best_weighted: dict[str, float] = {}

        for q in queries:
            extraction = extract_brand_from_query(q.query, self._brand_set)
            search_query = extraction.model_part or extraction.original_query
            if not search_query.strip():
                continue
            cands = self._embedding_retriever.retrieve(
                search_query,
                brand_filter=extraction.brand,
                top_k=self.config.top_k,
            )
            for c in cands:
                wscore = float(c.score) * float(q.weight)
                prev = best_weighted.get(c.canonical)
                if prev is None or wscore > prev:
                    best_weighted[c.canonical] = wscore
                    best[c.canonical] = RetrievedCandidate(
                        canonical=c.canonical,
                        score=wscore,
                        lexical_score=0.0,
                        semantic_score=c.score,
                    )

        merged = list(best.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    def _retrieve_for_queries(self, queries: list[LogQuery]) -> list[RetrievedCandidate]:
        """
        여러 query에 대해 retrieval을 수행하고 결과를 통합합니다.

        통합 방식:
        - canonical별로 최고(weighted) 점수를 유지
        - 최종 점수 = candidate.score * query.weight
        """
        if self._embedding_retriever is not None:
            return self._retrieve_accuracy(queries)

        best: dict[str, RetrievedCandidate] = {}
        best_weighted: dict[str, float] = {}

        for q in queries:
            cands = self._hybrid_retriever.retrieve(
                q.query,
                top_n_lexical=self.config.top_n_lexical,
                top_k=self.config.top_k,
            )
            for c in cands:
                wscore = float(c.score) * float(q.weight)
                prev = best_weighted.get(c.canonical)
                if prev is None or wscore > prev:
                    best_weighted[c.canonical] = wscore
                    best[c.canonical] = RetrievedCandidate(
                        canonical=c.canonical,
                        score=wscore,
                        lexical_score=float(c.lexical_score),
                        semantic_score=c.semantic_score,
                    )

        merged = list(best.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    def _apply_cosine_gate(
        self,
        pred: Prediction,
        primary_query: str,
        candidates: list[RetrievedCandidate],
    ) -> Prediction:
        """
        UNKNOWN이 아닌 예측에 대해 2단계 검증을 수행합니다:
        1) 어휘/서브스트링 중첩: primary_query와 예측 모델 간 중첩이 없으면 UNKNOWN
        2) 코사인 유사도: primary_query vs canonical 간 유사도가 임계치 미만이면 UNKNOWN
        3) top1이 거절되면 overlap+cosine 통과하는 다른 후보로 fallback (ARC ULTRA → ARCULTRA 등)

        사운드바가 아닌 기기(SKY, EE TV 등) 오탐을 줄이며, 정답이 top2~k에 있는 경우도 반영합니다.
        """
        # below_threshold/small_margin: top1 실패 시 cosine 통과하는 다른 후보 시도
        if pred.canonical_model is None and candidates and pred.evidence.get("reason") in (
            "below_threshold",
            "small_margin",
        ):
            pseudo = Prediction(
                canonical_model=candidates[0].canonical,
                confidence=pred.confidence,
                evidence=pred.evidence,
            )
            pred = self._apply_cosine_gate(pseudo, primary_query, candidates)

        if pred.canonical_model is None:
            return pred

        # generic_soundbar_rule(SOUNDBAR→ETC SOUNDBAR): 규칙 기반이므로 cosine gate로 덮어쓰지 않음
        if pred.evidence.get("reason") == "generic_soundbar_rule":
            return pred

        retriever = self._embedding_retriever or self._hybrid_retriever
        threshold = self.config.min_cosine_similarity
        min_lex = getattr(self.config, "min_lexical_similarity", None)

        # 질의 모델 파트 추출 (SONOS ARC ULTRA → ARCULTRA, canonical 모델 파트와 비교)
        extraction = extract_brand_from_query(primary_query, self._brand_set)
        q_model_part = (extraction.model_part or extraction.original_query or "").strip()
        q_compact = "".join(q_model_part.upper().split()) if q_model_part else ""

        # cosine 통과하는 후보 중 선택 (상위 N개 검사, 통과 시 렉시컬 재순위화)
        max_n = getattr(self.config, "cosine_gate_max_candidates", 5) or 5
        candidates_to_check = candidates[:max_n]

        # 배치 유사도 계산 (1~2회 encode로 N개 후보 처리, 속도 개선)
        passing: list[tuple[RetrievedCandidate, float]] = []
        if threshold is None or retriever is None:
            passing = [(c, 1.0) for c in candidates_to_check]
        else:
            canonicals_to_check = [c.canonical for c in candidates_to_check]
            try:
                batch_sims = (
                    retriever.compute_similarities_batch(primary_query, canonicals_to_check)
                    if hasattr(retriever, "compute_similarities_batch")
                    else [retriever.compute_similarity(primary_query, canon) for canon in canonicals_to_check]
                )
                if q_compact and hasattr(retriever, "compute_similarities_batch"):
                    mp_compacts = [
                        "".join((_get_canonical_model_part(c.canonical) or c.canonical or "").upper().split())
                        for c in candidates_to_check
                    ]
                    compact_sims = retriever.compute_similarities_batch(q_compact, mp_compacts)
                    batch_sims = [max(s, cs) for s, cs in zip(batch_sims, compact_sims)]
                for c, sim in zip(candidates_to_check, batch_sims):
                    mp_compact = "".join((_get_canonical_model_part(c.canonical) or c.canonical or "").upper().split())
                    if q_compact and mp_compact and q_compact == mp_compact:
                        passing.append((c, 1.0))
                        continue
                    if sim >= threshold:
                        passing.append((c, sim))
            except Exception:
                pass

        # 렉시컬 재순위화: cosine 동일 시 모델 파트 문자열 유사도로 우선순위
        def _lex_ratio(cand: RetrievedCandidate) -> float:
            mp = "".join(
                (_get_canonical_model_part(cand.canonical) or cand.canonical or "").upper().split()
            )
            return SequenceMatcher(None, q_compact, mp).ratio() if q_compact and mp else 0.0

        passing_with_lex = [(c, sim, _lex_ratio(c)) for c, sim in passing]
        passing_with_lex.sort(key=lambda x: (x[1], x[2]), reverse=True)  # cosine desc, lex desc
        best = (passing_with_lex[0][0], passing_with_lex[0][1], passing_with_lex[0][2]) if passing_with_lex else None

        # 렉시컬 거부 게이트: 모델 파트 유사도가 매우 낮으면 UNKNOWN (YAMAHA RX-A1040 vs SRB40 등)
        if best is not None and min_lex is not None and best[2] < min_lex:
            top1 = candidates[0] if candidates else None
            return Prediction(
                canonical_model=None,
                confidence=0.85,
                evidence={
                    "reason": "low_lexical_similarity",
                    "primary_query": primary_query,
                    "top1": top1.__dict__ if top1 else None,
                    "lexical_ratio": round(best[2], 4),
                    "threshold": min_lex,
                },
            )

        if best is not None:
            c, sim = best[0], best[1]
            return Prediction(
                canonical_model=c.canonical,
                confidence=float(min(1.0, max(0.0, c.score))),
                evidence={
                    "reason": "cosine_gate_accepted",
                    "primary_query": primary_query,
                    "selected": c.__dict__,
                    "cosine_similarity": round(sim, 4) if sim > 0 else None,
                },
            )

        # 모든 후보가 통과 실패 → UNKNOWN (cosine 기반)
        top1 = candidates[0] if candidates else None
        reason = "low_cosine_similarity"
        first_sim = 0.0
        if top1 and retriever and threshold is not None:
            try:
                first_sim = retriever.compute_similarity(primary_query, top1.canonical)
            except Exception:
                pass
        return Prediction(
            canonical_model=None,
            confidence=0.85,
            evidence={
                "reason": reason,
                "primary_query": primary_query,
                "top1": top1.__dict__ if top1 else None,
                "cosine_similarity": round(first_sim, 4) if first_sim > 0 else None,
                "threshold": threshold,
            },
        )

    def predict_for_query(self, typed: TypedQuery) -> PredictionResult:
        """
        단일 TypedQuery에 대해 사운드바 모델을 예측합니다.

        Args:
            typed: TypedQuery (type_, source, query, raw, weight)

        Returns:
            PredictionResult
        """
        log_q = LogQuery(query=typed.query, raw=typed.raw, source=typed.source, weight=typed.weight)
        if self._embedding_retriever is not None:
            extraction = extract_brand_from_query(typed.query, self._brand_set)
            search_query = extraction.model_part or extraction.original_query
            cands = self._embedding_retriever.retrieve(
                search_query,
                brand_filter=extraction.brand,
                top_k=self.config.top_k,
            )
            weighted = self._embedding_cands_to_retrieved(cands, typed.weight)
        else:
            cands = self._hybrid_retriever.retrieve(
                typed.query,
                top_n_lexical=self.config.top_n_lexical,
                top_k=self.config.top_k,
            )
            weighted = [
                RetrievedCandidate(
                    canonical=c.canonical,
                    score=float(c.score) * float(typed.weight),
                    lexical_score=float(c.lexical_score) if c.lexical_score is not None else 0.0,
                    semantic_score=c.semantic_score,
                )
                for c in cands
            ]
        pred: Prediction = self._verifier.verify([log_q], weighted)
        # small_margin이면 verifier가 불확실한 것 → cosine gate로 후보 재검토
        if (
            pred.canonical_model is None
            and pred.evidence.get("reason") == "small_margin"
            and weighted
        ):
            pseudo = Prediction(
                canonical_model=weighted[0].canonical,
                confidence=pred.confidence,
                evidence=pred.evidence,
            )
            pred = self._apply_cosine_gate(pseudo, typed.query, weighted)
        else:
            pred = self._apply_cosine_gate(pred, typed.query, weighted)
        return PredictionResult(
            row_id=None,
            predicted=pred.canonical_model,
            confidence=float(pred.confidence),
            primary_query=typed.raw,
            candidates=weighted,
            evidence=dict(pred.evidence),
        )

    def predict_row(self, row: dict[str, Any], *, row_id: Optional[int] = None) -> PredictionResult:
        """
        단일 로그 row(dict)에 대해 사운드바 모델을 예측합니다.
        (기존 통합 방식: 모든 소스 후보를 합쳐 1개 예측)

        Args:
            row: CSV row를 dict로 변환한 값
            row_id: 행 식별자(선택)

        Returns:
            PredictionResult
        """
        queries = build_log_queries_from_row(
            row,
            soundbar_brand_set=self._brand_set,
            include_bt=self.config.include_bt,
        )
        primary_log = choose_primary_query(queries)
        primary = primary_log.query if primary_log else (queries[0].query if queries else "")
        candidates = self._retrieve_for_queries(queries)
        pred: Prediction = self._verifier.verify(queries, candidates)
        pred = self._apply_cosine_gate(pred, primary, candidates)
        out_primary = primary_log.raw if primary_log else primary
        return PredictionResult(
            row_id=row_id,
            predicted=pred.canonical_model,
            confidence=float(pred.confidence),
            primary_query=out_primary,
            candidates=candidates,
            evidence=dict(pred.evidence),
        )

    def batch_predict(self, df: pd.DataFrame, *, row_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        여러 로그 row를 일괄 예측합니다 (기존 통합 방식: 행당 1개 예측).

        Args:
            df: 로그 DataFrame
            row_id_col: row id 컬럼명이 있으면 이를 사용(없으면 인덱스 사용)

        Returns:
            예측 결과 DataFrame
        """
        rows: list[dict[str, Any]] = df.to_dict(orient="records")
        out_rows: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            rid = i
            if row_id_col and row_id_col in df.columns:
                try:
                    rid = int(r.get(row_id_col))
                except Exception:
                    rid = i

            res = self.predict_row(r, row_id=rid)
            top1_score = res.candidates[0].score if res.candidates else None
            top5 = [c.canonical for c in res.candidates[:5]]
            top5_candidates = [c.__dict__ for c in res.candidates[:5]]
            out_rows.append(
                {
                    "row_id": rid,
                    "predicted_model": res.predicted if res.predicted is not None else "UNKNOWN",
                    "true_model": "",
                    "confidence": res.confidence,
                    "primary_query": res.primary_query,
                    "top1_score": top1_score,
                    "top5": json_dumps_safe(top5),
                    "top5_candidates": json_dumps_safe(top5_candidates),
                    "evidence": json_dumps_safe(res.evidence),
                }
            )
        return pd.DataFrame(out_rows)

    def batch_predict_per_source(self, df: pd.DataFrame, *, row_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        여러 로그 row를 소스별(BT/HDMI)로 독립 예측합니다.
        행당 최대 5개(BT 1 + HDMI 4) 예측 결과를 반환합니다.

        Args:
            df: 로그 DataFrame
            row_id_col: row id 컬럼명이 있으면 이를 사용(없으면 인덱스 사용)

        Returns:
            예측 결과 DataFrame (row_id, type, predicted_model, true_model, primary_query 등)
        """
        rows: list[dict[str, Any]] = df.to_dict(orient="records")
        out_rows: list[dict[str, Any]] = []
        out_id = 0
        for i, r in enumerate(rows):
            rid = i
            if row_id_col and row_id_col in df.columns:
                try:
                    rid = int(r.get(row_id_col))
                except Exception:
                    rid = i

            typeds = build_typed_queries_from_row(
                r,
                soundbar_brand_set=self._brand_set,
                include_bt=self.config.include_bt,
            )
            for tq in typeds:
                res = self.predict_for_query(tq)
                top1_score = res.candidates[0].score if res.candidates else None
                top5 = [c.canonical for c in res.candidates[:5]]
                top5_candidates = [c.__dict__ for c in res.candidates[:5]]
                out_rows.append(
                    {
                        "id": out_id,
                        "row_id": rid,
                        "type": tq.type_,
                        "predicted_model": res.predicted if res.predicted is not None else "UNKNOWN",
                        "true_model": "",
                        "confidence": res.confidence,
                        "primary_query": res.primary_query,
                        "top1_score": top1_score,
                        "top5": json_dumps_safe(top5),
                        "top5_candidates": json_dumps_safe(top5_candidates),
                        "evidence": json_dumps_safe(res.evidence),
                    }
                )
                out_id += 1
        return pd.DataFrame(out_rows)


def json_dumps_safe(obj: Any) -> str:
    """
    JSON 직렬화를 안전하게 수행합니다.

    Args:
        obj: 임의 객체

    Returns:
        JSON 문자열(실패 시 repr)
    """
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return repr(obj)

