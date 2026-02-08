"""
텍스트 정규화 유틸리티 모듈입니다.

본 프로젝트는 로그 문자열과 DB 모델 문자열 간의 매칭을 수행하므로,
대소문자/특수문자/공백 등 표면적인 차이를 최소화하는 정규화가 중요합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


_WS_RE = re.compile(r"\s+")
_BRACKET_RE = re.compile(r"\([^)]*\)")
_KEEP_CHARS_RE = re.compile(r"[^A-Z0-9\.\+/\- ]+")


@dataclass(frozen=True)
class NormalizationConfig:
    """정규화 동작을 제어하는 설정값입니다."""

    drop_bracketed: bool = True
    keep_dot_plus_slash_dash: bool = True
    uppercase: bool = True
    collapse_whitespace: bool = True
    remove_hyphen: bool = True  # HT-S40R -> HTS40R 매칭 개선


def normalize_text(text: Optional[str], config: Optional[NormalizationConfig] = None) -> str:
    """
    주어진 문자열을 매칭 친화적으로 정규화합니다.

    - 대문자 변환
    - 괄호(...) 제거(옵션)
    - 영숫자/일부 기호(. + / -) 이외 제거
    - 공백 정리

    Args:
        text: 입력 텍스트 (None 가능).
        config: 정규화 설정. None이면 기본값 사용.

    Returns:
        정규화된 문자열(빈 문자열 가능).
    """
    if not text:
        return ""
    cfg = config or NormalizationConfig()

    s = text.strip()
    if cfg.uppercase:
        s = s.upper()
    if cfg.drop_bracketed:
        # 예: "LG SQC2(CC)" -> "LG SQC2"
        s = _BRACKET_RE.sub(" ", s)

    if cfg.keep_dot_plus_slash_dash:
        s = _KEEP_CHARS_RE.sub(" ", s)
    else:
        s = re.sub(r"[^A-Z0-9 ]+", " ", s)

    if cfg.collapse_whitespace:
        s = _WS_RE.sub(" ", s).strip()
    if cfg.remove_hyphen:
        s = s.replace("-", "").strip()
        if cfg.collapse_whitespace:
            s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_brand(brand: Optional[str]) -> str:
    """
    브랜드 문자열을 정규화합니다.

    Args:
        brand: 브랜드(제조사) 문자열.

    Returns:
        정규화된 브랜드 문자열(대문자, 공백 정리).
    """
    return normalize_text(brand)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """
    입력 시퀀스에서 중복을 제거하되, 최초 등장 순서를 유지합니다.

    Args:
        items: 문자열 iterable.

    Returns:
        중복 제거된 리스트.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

