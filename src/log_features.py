"""
로그 행에서 사운드바 후보 문자열(query)을 구성하고, placeholder/노이즈를 필터링합니다.

`raw_data/HDMI_BT_Log.csv`는 여러 입력 소스(NAME1~4)와 브랜드(BRAND1~4),
그리고 BT 장치명(NAME_BT)을 포함합니다. 여기서 실제 사운드바 모델 매칭에
의미있는 후보 문자열을 추출하는 것이 목적입니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .normalize import normalize_brand, normalize_text, unique_preserve_order
import math


_HDMI_RE = re.compile(r"^HDMI(\s*\d+)?$")
_PLACEHOLDER_EXACT = {
    "AV",
    "AUX",
    "USB",
    "OPTICAL",
    "DIGITAL",
    "DIGITAL IN",
    "TV",
    "TV IN",
    "IN",
    "HDMI",
}

_BT_HEADPHONE_HINT_RE = re.compile(
    r"\b(AIRPODS|EARBUD|EARBUDS|HEADPHONE|HEADPHONES|TUNE|TOUR|LIVE|WH-|WF-)\b"
)


@dataclass(frozen=True)
class LogQuery:
    """
    로그에서 추출한 후보 검색 질의 문자열입니다.

    Attributes:
        query: 정규화된 검색 질의
        raw: 원본 문자열(가능하면 유지)
        source: 어떤 필드에서 왔는지(예: NAME2, NAME_BT 등)
        weight: 이후 검색/검증에서 가중치로 사용할 값(기본 1.0)
    """

    query: str
    raw: str
    source: str
    weight: float = 1.0


def is_placeholder(text: str) -> bool:
    """
    HDMI/AV 같은 placeholder 입력 소스명을 탐지합니다.

    주의: \"APPLE TV\" 같은 정상 장치명은 placeholder로 처리하지 않도록
    짧은 토큰/정규식 기반으로만 필터합니다.

    Args:
        text: 정규화된 문자열(권장: `normalize_text` 결과)

    Returns:
        placeholder로 판단되면 True
    """
    if not text:
        return True
    s = text.strip().upper()
    if _HDMI_RE.match(s):
        return True
    if s in _PLACEHOLDER_EXACT:
        return True
    # 매우 짧고 의미없는 입력은 제외
    if len(s) <= 2:
        return True
    return False


def bt_looks_like_non_soundbar(bt_name: str) -> bool:
    """
    BT 장치명이 헤드폰/이어폰 등 사운드바가 아닐 가능성이 높은지 휴리스틱으로 판단합니다.

    Args:
        bt_name: 정규화된 BT 이름

    Returns:
        사운드바가 아닐 가능성이 높으면 True
    """
    if not bt_name:
        return False
    s = bt_name.upper()
    return _BT_HEADPHONE_HINT_RE.search(s) is not None


def build_log_queries_from_row(
    row: dict[str, Any],
    soundbar_brand_set: Optional[set[str]] = None,
    include_bt: bool = True,
) -> list[LogQuery]:
    """
    단일 로그 row에서 후보 질의 문자열 리스트를 생성합니다.

    생성 규칙(요약):
    - NAME1~4에서 placeholder 제외 후 후보 생성
    - BRANDi가 있으면 \"BRANDi NAMEi\" 형태 후보를 추가(브랜드 강화)
    - NAME_BT는 노이즈(헤드폰/이어폰) 가능성이 높아 기본 weight를 낮춰 추가(옵션)

    Args:
        row: pandas row를 `to_dict()`한 형태를 권장.
        soundbar_brand_set: 사운드바 DB 기반 브랜드 집합(정규화된 값).
            제공되면 브랜드가 DB에 있는 경우 가중치/우선순위를 강화합니다.
        include_bt: NAME_BT를 후보로 포함할지 여부.

    Returns:
        LogQuery 리스트(중복 제거, 순서 유지).
    """
    candidates: list[LogQuery] = []

    def _is_missing(x: Any) -> bool:
        """
        pandas NaN 등 결측값을 탐지합니다.

        주의:
        - CSV/전처리 과정에 따라 결측이 float NaN이 아니라 문자열 "nan"/"NaN"으로 들어올 수 있습니다.
          이 경우 query가 "NAN CHROMECAST"처럼 오염되어 매칭 오탐을 유발할 수 있어,
          문자열 결측도 결측으로 처리합니다.
        """
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"nan", "none", "null", ""}:
                return True
        return False

    def _add(query_raw: str, source: str, weight: float) -> None:
        q = normalize_text(query_raw)
        if not q or is_placeholder(q):
            return
        candidates.append(LogQuery(query=q, raw=str(query_raw), source=source, weight=weight))

    # NAME1~4 / BRAND1~4
    for i in range(1, 5):
        name_key = f"NAME{i}"
        brand_key = f"BRAND{i}"
        name_raw = row.get(name_key, "")
        brand_raw = row.get(brand_key, "")

        name = "" if _is_missing(name_raw) else normalize_text(str(name_raw))
        brand = "" if _is_missing(brand_raw) else normalize_brand(str(brand_raw))

        if name and not is_placeholder(name):
            _add(str(name_raw), name_key, weight=1.0)

            if brand:
                w = 1.1
                if soundbar_brand_set and brand in soundbar_brand_set:
                    w = 1.25
                # 모델명에 이미 브랜드가 있으면 모델명만, 없으면 브랜드+모델명
                if name.upper().startswith(brand.upper()):
                    brand_name_raw = str(name_raw).strip()
                else:
                    brand_name_raw = f"{str(brand_raw).strip()} {str(name_raw).strip()}".strip()
                _add(brand_name_raw, f"{brand_key}+{name_key}", weight=w)

    # NAME_BT (옵션)
    if include_bt:
        bt_raw = row.get("NAME_BT", "")
        bt = "" if _is_missing(bt_raw) else normalize_text(str(bt_raw))
        if bt and not is_placeholder(bt):
            weight = 0.63  # 0.6 → 0.63 (5% 상향, below_threshold 완화)
            if bt_looks_like_non_soundbar(bt):
                weight = 0.35
            _add(str(bt_raw), "NAME_BT", weight=weight)

    # 중복 제거(정규화 query 기준)
    ordered_queries = unique_preserve_order([c.query for c in candidates])
    first_by_query: dict[str, LogQuery] = {}
    for c in candidates:
        if c.query not in first_by_query:
            first_by_query[c.query] = c

    return [first_by_query[q] for q in ordered_queries if q in first_by_query]


@dataclass(frozen=True)
class TypedQuery:
    """
    소스별(타입별) 검색 질의입니다. BT/HDMI 각각 독립 예측을 위해 사용합니다.

    Attributes:
        type_: "BT" 또는 "HDMI"
        source: 필드명(NAME_BT, NAME1, NAME2, NAME3, NAME4)
        query: 정규화된 검색 질의
        raw: 원본 문자열(primary_query 출력용)
        weight: 가중치
    """

    type_: str
    source: str
    query: str
    raw: str
    weight: float = 1.0


def build_typed_queries_from_row(
    row: dict[str, Any],
    soundbar_brand_set: Optional[set[str]] = None,
    include_bt: bool = True,
) -> list[TypedQuery]:
    """
    단일 로그 row에서 소스별(BT/HDMI) 질의 리스트를 생성합니다.

    - BT: NAME_BT가 유효하면 1개
    - HDMI: (NAME1,BRAND1)~(NAME4,BRAND4) 각 쌍이 유효하면 1개씩, 최대 4개

    각 소스별로 독립 예측을 수행할 때 사용합니다.

    Args:
        row: pandas row를 dict로 변환한 값
        soundbar_brand_set: 사운드바 DB 브랜드 집합
        include_bt: NAME_BT 포함 여부

    Returns:
        TypedQuery 리스트(소스별 1개, 중복 없음)
    """
    result: list[TypedQuery] = []

    def _is_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"nan", "none", "null", ""}:
                return True
        return False

    # BT: NAME_BT
    if include_bt:
        bt_raw = row.get("NAME_BT", "")
        if not _is_missing(bt_raw):
            bt = normalize_text(str(bt_raw))
            if bt and not is_placeholder(bt):
                w = 0.6 if bt_looks_like_non_soundbar(bt) else 0.6
                result.append(
                    TypedQuery(type_="BT", source="NAME_BT", query=bt, raw=str(bt_raw).strip(), weight=w)
                )

    # HDMI: (NAME1,BRAND1) ~ (NAME4,BRAND4)
    for i in range(1, 5):
        name_key = f"NAME{i}"
        brand_key = f"BRAND{i}"
        name_raw = row.get(name_key, "")
        brand_raw = row.get(brand_key, "")

        name = "" if _is_missing(name_raw) else normalize_text(str(name_raw))
        brand = "" if _is_missing(brand_raw) else normalize_brand(str(brand_raw))

        if not name or is_placeholder(name):
            continue

        # primary_query 출력: 모델명에 브랜드가 있으면 모델명만, 없으면 브랜드+모델명
        if brand:
            raw = (
                str(name_raw).strip()
                if name.upper().startswith(brand.upper())
                else f"{str(brand_raw).strip()} {str(name_raw).strip()}".strip()
            )
        else:
            raw = str(name_raw).strip()
        if brand and not (name.upper().startswith(brand.upper())):
            query = normalize_text(f"{str(brand_raw).strip()} {str(name_raw).strip()}")
            w = 1.1
            if soundbar_brand_set and brand in soundbar_brand_set:
                w = 1.25
        else:
            query = name
            w = 1.0

        result.append(TypedQuery(type_="HDMI", source=name_key, query=query, raw=raw, weight=w))

    return result


def choose_primary_query(queries: Iterable[LogQuery]) -> Optional[LogQuery]:
    """
    여러 후보 LogQuery 중 1개를 대표 질의로 선택합니다.

    현재는 가장 weight가 크고, 길이가 적당히 긴 후보를 우선합니다.

    Args:
        queries: LogQuery iterable

    Returns:
        선택된 LogQuery 또는 후보가 없으면 None
    """
    qs = list(queries)
    if not qs:
        return None
    return sorted(qs, key=lambda q: (q.weight, len(q.query)), reverse=True)[0]

