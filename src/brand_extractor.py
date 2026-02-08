"""
기기명에서 브랜드(Brand)를 추출하는 모듈입니다.

사운드바 DB의 known brand 집합을 사용하여 dictionary 기반 추출을 수행합니다.
(일반 NER 대신 도메인 특화 방식으로 정확도를 높입니다.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from .normalize import normalize_text


@dataclass(frozen=True)
class BrandExtraction:
    """
    브랜드 추출 결과입니다.

    Attributes:
        brand: 추출된 브랜드(정규화된 문자열). 없으면 None.
        model_part: 브랜드 제외 나머지 부분(정규화). 없으면 query 전체.
        original_query: 원본 질의.
    """

    brand: Optional[str]
    model_part: str
    original_query: str


def extract_brand_from_query(
    query: str,
    known_brands: Set[str],
) -> BrandExtraction:
    """
    질의 문자열에서 브랜드를 추출합니다.

    전략:
    1. 정규화된 query의 첫 토큰이 known_brands에 있으면 brand로 사용.
    2. 첫 토큰이 없으면, query 내 임의 위치에서 가장 긴 brand 매칭을 탐색.
    3. brand가 없으면 model_part = query 전체.

    Args:
        query: 기기명 질의(원본 또는 정규화).
        known_brands: 사운드바 DB의 브랜드 집합(정규화된 값).

    Returns:
        BrandExtraction
    """
    if not query or not query.strip():
        return BrandExtraction(brand=None, model_part="", original_query=query or "")

    q = normalize_text(query)
    if not q:
        return BrandExtraction(brand=None, model_part="", original_query=query)

    tokens = q.split()
    if not tokens:
        return BrandExtraction(brand=None, model_part=q, original_query=query)

    # 1) 첫 토큰이 브랜드인 경우
    first = tokens[0]
    if first in known_brands:
        model_part = " ".join(tokens[1:]).strip()
        return BrandExtraction(brand=first, model_part=model_part or q, original_query=query)

    # 2) 쿼리 내에서 가장 긴 브랜드 매칭 탐색
    matched_brand: Optional[str] = None
    matched_len = 0
    for b in known_brands:
        if not b:
            continue
        # "LG" in "LG SPEAKER DS80TR" 또는 "SONOS" in "SONOS ARC"
        if b in q:
            if len(b) > matched_len:
                matched_len = len(b)
                matched_brand = b

    if matched_brand:
        # 브랜드 제거 후 model_part
        rest = q.replace(matched_brand, "", 1).strip()
        rest = " ".join(rest.split())
        return BrandExtraction(brand=matched_brand, model_part=rest or q, original_query=query)

    # 3) 브랜드 없음
    return BrandExtraction(brand=None, model_part=q, original_query=query)
