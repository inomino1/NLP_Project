"""
룰/Levenshtein 기반 baseline 매핑 모듈입니다.

이 모듈은 비교용(베이스라인)으로 다음 파이프라인을 제공합니다.

- 사운드바 DB 로드: `soundbar_list.py`의 `data = [...]` 파싱(기존 `soundbar_db` 재사용)
- 로그 질의 생성: `NAME1~4`, `BRAND1~4`, (옵션) `NAME_BT`로부터 후보 query 생성(기존 `log_features` 재사용)
- 문자열 유사도: Levenshtein distance 기반 유사도(0~1)를 계산하여 최고 점수 모델을 선택
- UNKNOWN 결정: 최고 점수가 임계치 미만이면 UNKNOWN(None) 반환

주의:
- 정확도는 최신 임베딩/재랭킹 파이프라인 대비 낮을 수 있으며, 이 모듈은
  “과거/단순 규칙 기반 방식”과의 비교 기준을 제공하는 것이 목적입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from difflib import SequenceMatcher

from .log_features import LogQuery, TypedQuery, build_log_queries_from_row, build_typed_queries_from_row
from .normalize import normalize_text
from .soundbar_db import SoundbarRecord, get_brand_set, load_soundbar_db_from_py


@dataclass(frozen=True)
class BaselineConfig:
    """Levenshtein baseline 하이퍼파라미터 설정입니다."""

    include_bt: bool = True
    accept_score_threshold: float = 0.68
    top_k: int = 5
    use_brand_filter: bool = True
    prefilter_top_n: int = 250


@dataclass(frozen=True)
class BaselineCandidate:
    """
    baseline 후보 1개를 나타냅니다.

    Attributes:
        canonical: 표준 canonical 모델명(예: "LG S90TY")
        score: 최종 점수(0~1), query weight 반영
        raw_score: weight 미반영 유사도(0~1)
        query: 사용된 query 문자열(정규화)
        source: query의 출처 필드명
    """

    canonical: str
    score: float
    raw_score: float
    query: str
    source: str


@dataclass(frozen=True)
class BaselinePredictionResult:
    """
    baseline의 단일 row 예측 결과입니다.

    Attributes:
        row_id: 원본 데이터 행 식별자
        predicted: 예측 canonical 모델명(UNKNOWN이면 None)
        confidence: 신뢰도(여기서는 top1 score 사용)
        primary_query: 대표 질의(정규화 문자열)
        candidates: 상위 후보 리스트(내림차순)
        evidence: 간단 근거 dict
    """

    row_id: Optional[int]
    predicted: Optional[str]
    confidence: float
    primary_query: str
    candidates: list[BaselineCandidate]
    evidence: dict[str, Any]


def levenshtein_distance(a: str, b: str) -> int:
    """
    Levenshtein distance(편집 거리)를 계산합니다.

    Args:
        a: 문자열 A
        b: 문자열 B

    Returns:
        a를 b로 바꾸는 최소 편집 연산 횟수
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # O(min(n,m)) 메모리 DP
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """
    Levenshtein distance를 0~1 유사도로 변환합니다.

    유사도는 \(1 - dist / max_len\)로 정의합니다.

    Args:
        a: 문자열 A (정규화된 문자열을 권장)
        b: 문자열 B (정규화된 문자열을 권장)

    Returns:
        0~1 유사도. 둘 다 빈 문자열이면 1.0
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len <= 0:
        return 0.0
    d = levenshtein_distance(a, b)
    return max(0.0, 1.0 - (float(d) / float(max_len)))


class BaselineSoundbarMatcher:
    """
    룰/Levenshtein 기반 baseline 매처입니다.

    Public API:
    - `predict_row(row_dict, row_id=None) -> BaselinePredictionResult`
    - `batch_predict(df) -> pd.DataFrame`
    """

    def __init__(self, soundbar_list_py: Path, config: Optional[BaselineConfig] = None) -> None:
        """
        Args:
            soundbar_list_py: `soundbar_list.py` 경로
            config: BaselineConfig
        """
        self.soundbar_list_py = soundbar_list_py
        self.config = config or BaselineConfig()

        self._records: list[SoundbarRecord] = []
        self._brand_set: set[str] = set()
        self._canonicals: list[str] = []
        self._canonical_by_brand: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        """사운드바 DB를 로드하고 브랜드 인덱스를 준비합니다."""
        self._records = load_soundbar_db_from_py(self.soundbar_list_py)
        self._brand_set = get_brand_set(self._records)
        self._canonicals = [r.canonical for r in self._records]

        by_brand: dict[str, list[str]] = {}
        for r in self._records:
            by_brand.setdefault(r.brand, []).append(r.canonical)
        self._canonical_by_brand = by_brand

    @property
    def brand_set(self) -> set[str]:
        """DB로부터 추출된 브랜드 집합을 반환합니다."""
        return set(self._brand_set)

    def _extract_brand_hint(self, query: str) -> Optional[str]:
        """
        query 문자열에서 브랜드 힌트를 추출합니다.

        Args:
            query: 정규화된 query

        Returns:
            브랜드 문자열 또는 None
        """
        q = normalize_text(query)
        if not q:
            return None

        # 첫 토큰이 브랜드인 경우가 많음
        first = q.split()[0] if q.split() else ""
        if first and first in self._brand_set:
            return first

        # 부분 포함(예: "SAMSUNG HW-Q..." 내 "SAMSUNG")
        for b in self._brand_set:
            if b and b in q:
                return b
        return None

    def _candidate_pool_for_query(self, query: str) -> list[str]:
        """
        query에 대해 비교할 canonical 후보 풀을 선택합니다.

        기본은 전체 canonical을 비교하되, `use_brand_filter=True`이고 브랜드 힌트가 있으면
        동일 브랜드 모델만으로 후보 풀을 축소합니다.
        """
        if not self.config.use_brand_filter:
            return self._canonicals
        b = self._extract_brand_hint(query)
        if not b:
            return self._canonicals
        return self._canonical_by_brand.get(b, self._canonicals)

    def _prefilter_pool(self, query: str, pool: list[str]) -> list[str]:
        """
        Levenshtein 계산 전에 후보 풀을 값싼 휴리스틱으로 축소합니다.

        - `SequenceMatcher.ratio()`는 Levenshtein보다 훨씬 빠르며,
          전수 비교에서 후보를 상위 N개로 줄이는 데 사용합니다.

        Args:
            query: 정규화된 query
            pool: 비교 대상 canonical 풀

        Returns:
            축소된 canonical 리스트(최대 `prefilter_top_n`)
        """
        n = int(self.config.prefilter_top_n)
        if n <= 0 or len(pool) <= n:
            return pool

        q = normalize_text(query)
        scored: list[tuple[float, str]] = []
        for c in pool:
            # ratio는 0~1
            r = SequenceMatcher(None, q, c).ratio()
            scored.append((float(r), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:n]]

    def _score_one_query(self, q: LogQuery) -> list[BaselineCandidate]:
        """
        단일 query에 대해 DB 전체(또는 필터된 풀)와 유사도를 계산하여 후보를 반환합니다.

        Args:
            q: LogQuery

        Returns:
            BaselineCandidate 리스트(내림차순)
        """
        pool0 = self._candidate_pool_for_query(q.query)
        pool = self._prefilter_pool(q.query, pool0)
        out: list[BaselineCandidate] = []
        for canonical in pool:
            raw = levenshtein_similarity(q.query, canonical)
            score = float(raw) * float(q.weight)
            out.append(
                BaselineCandidate(
                    canonical=canonical,
                    score=score,
                    raw_score=float(raw),
                    query=q.query,
                    source=q.source,
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[: max(1, int(self.config.top_k))]

    def predict_for_typed_query(self, typed: TypedQuery) -> BaselinePredictionResult:
        """
        단일 TypedQuery에 대해 사운드바 모델을 예측합니다.

        Args:
            typed: TypedQuery (type_, source, query, raw, weight)

        Returns:
            BaselinePredictionResult
        """
        log_q = LogQuery(query=typed.query, raw=typed.raw, source=typed.source, weight=typed.weight)
        merged = self._score_one_query(log_q)
        top1 = merged[0] if merged else None
        predicted: Optional[str] = None
        conf = float(top1.score) if top1 else 0.0
        if top1 and top1.score >= float(self.config.accept_score_threshold):
            predicted = top1.canonical
        
        # reason 결정
        if not top1:
            reason = "no_candidates"
        elif predicted is None:
            reason = f"score_below_threshold (top1_score={top1.score:.4f} < threshold={self.config.accept_score_threshold})"
        else:
            reason = f"score_above_threshold (top1_score={top1.score:.4f} >= threshold={self.config.accept_score_threshold})"
        
        evidence: dict[str, Any] = {
            "reason": reason,
            "primary_query": typed.raw,
            "method": "baseline_levenshtein",
            "accept_score_threshold": float(self.config.accept_score_threshold),
        }
        
        if top1:
            evidence.update(
                {
                    "top1_canonical": top1.canonical,
                    "top1_score": float(top1.score),
                    "top1_raw_score": float(top1.raw_score),
                }
            )
        return BaselinePredictionResult(
            row_id=None,
            predicted=predicted,
            confidence=conf,
            primary_query=typed.raw,
            candidates=merged,
            evidence=evidence,
        )

    def predict_row(self, row: dict[str, Any], *, row_id: Optional[int] = None) -> BaselinePredictionResult:
        """
        단일 로그 row(dict)에 대해 사운드바 모델을 예측합니다.

        Args:
            row: CSV row를 dict로 변환한 값
            row_id: 행 식별자(선택)

        Returns:
            BaselinePredictionResult
        """
        queries = build_log_queries_from_row(
            row,
            soundbar_brand_set=self._brand_set,
            include_bt=self.config.include_bt,
        )
        primary = queries[0].query if queries else ""

        # query별 top_k 후보를 계산한 뒤, canonical 단위로 최고 점수만 유지
        best: dict[str, BaselineCandidate] = {}
        for q in queries:
            for cand in self._score_one_query(q):
                prev = best.get(cand.canonical)
                if prev is None or cand.score > prev.score:
                    best[cand.canonical] = cand

        merged = list(best.values())
        merged.sort(key=lambda c: c.score, reverse=True)
        topk = merged[: max(1, int(self.config.top_k))]

        top1 = topk[0] if topk else None
        predicted: Optional[str] = None
        conf = float(top1.score) if top1 else 0.0
        if top1 and top1.score >= float(self.config.accept_score_threshold):
            predicted = top1.canonical

        # reason 결정
        if not top1:
            reason = "no_candidates"
        elif predicted is None:
            reason = f"score_below_threshold (top1_score={top1.score:.4f} < threshold={self.config.accept_score_threshold})"
        else:
            reason = f"score_above_threshold (top1_score={top1.score:.4f} >= threshold={self.config.accept_score_threshold})"

        evidence: dict[str, Any] = {
            "reason": reason,
            "primary_query": primary,
            "method": "baseline_levenshtein",
            "accept_score_threshold": float(self.config.accept_score_threshold),
            "include_bt": bool(self.config.include_bt),
            "use_brand_filter": bool(self.config.use_brand_filter),
        }
        
        if top1:
            evidence.update(
                {
                    "top1_canonical": top1.canonical,
                    "top1_score": float(top1.score),
                    "top1_raw_score": float(top1.raw_score),
                    "top1_query": top1.query,
                    "top1_source": top1.source,
                }
            )

        return BaselinePredictionResult(
            row_id=row_id,
            predicted=predicted,
            confidence=conf,
            primary_query=primary,
            candidates=topk,
            evidence=evidence,
        )

    def batch_predict(self, df: pd.DataFrame, *, row_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        여러 로그 row를 일괄 예측합니다.

        Args:
            df: 로그 DataFrame
            row_id_col: row id 컬럼명이 있으면 이를 사용(없으면 인덱스 사용)

        Returns:
            예측 결과 DataFrame (agent 출력 컬럼과 호환되는 형태)
        """
        rows: list[dict[str, Any]] = df.to_dict(orient="records")
        out_rows: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            rid: Optional[int]
            if row_id_col and row_id_col in df.columns:
                try:
                    rid = int(r.get(row_id_col))
                except Exception:
                    rid = i
            else:
                rid = i

            res = self.predict_row(r, row_id=rid)
            top1_score = res.candidates[0].score if res.candidates else None
            top5 = [c.canonical for c in res.candidates[:5]]
            top5_candidates = [c.__dict__ for c in res.candidates[:5]]
            out_rows.append(
                {
                    "row_id": res.row_id,
                    "predicted_model": res.predicted if res.predicted is not None else "UNKNOWN",
                    "true_model": "",
                    "confidence": float(res.confidence),
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
            예측 결과 DataFrame (id, row_id, type, predicted_model, true_model, primary_query 등)
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
                include_bt=bool(self.config.include_bt),
            )
            for tq in typeds:
                res = self.predict_for_typed_query(tq)
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
                        "confidence": float(res.confidence),
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

