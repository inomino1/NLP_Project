"""
사운드바 표준 모델 DB 로드/파싱/정규화 모듈입니다.

현재 워크스페이스의 `soundbar_list.py`는 `data = [...]` 형태로 모델 목록을 보유합니다.
각 항목은 아래 구조(공백 구분 토큰)로 가정합니다.

- 첫 토큰: Brand
- 마지막 3토큰: Grade, Year, supportDolbyAtmos (혹은 'null')
- 중간 토큰: Model (공백이 있을 수 있으므로 join)

파일 입출력은 오류 처리를 포함합니다.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .normalize import normalize_brand, normalize_text, unique_preserve_order


@dataclass(frozen=True)
class SoundbarRecord:
    """사운드바 DB의 단일 레코드입니다."""

    brand: str
    model: str
    grade: Optional[str]
    year: Optional[int]
    support_dolby_atmos: Optional[bool]
    canonical: str


def _parse_optional_int(token: str) -> Optional[int]:
    """문자열 토큰을 int로 변환합니다. 실패 시 None을 반환합니다."""
    token = token.strip()
    if not token or token.lower() == "null":
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _parse_optional_bool(token: str) -> Optional[bool]:
    """문자열 토큰을 bool로 변환합니다. 실패 시 None을 반환합니다."""
    token = token.strip()
    if not token or token.lower() == "null":
        return None
    if token.upper() in {"O", "Y", "YES", "TRUE", "T"}:
        return True
    if token.upper() in {"X", "N", "NO", "FALSE", "F"}:
        return False
    return None


def parse_soundbar_item(item: str) -> Optional[SoundbarRecord]:
    """
    `soundbar_list.py`의 data 항목(문자열 1개)을 파싱합니다.

    Args:
        item: 예) "LG S90TY High 2024 O"

    Returns:
        SoundbarRecord 또는 파싱 실패 시 None.
    """
    if not item or not item.strip():
        return None
    tokens = item.split()
    if len(tokens) < 2:
        return None

    brand_raw = tokens[0]
    brand = normalize_brand(brand_raw)

    grade: Optional[str] = None
    year: Optional[int] = None
    atmos: Optional[bool] = None
    model_tokens: list[str] = tokens[1:]

    # 최소 5토큰 이상이면 마지막 3토큰을 메타로 간주
    if len(tokens) >= 5:
        grade_token = tokens[-3]
        year_token = tokens[-2]
        atmos_token = tokens[-1]
        grade = None if grade_token.lower() == "null" else normalize_text(grade_token)
        year = _parse_optional_int(year_token)
        atmos = _parse_optional_bool(atmos_token)
        model_tokens = tokens[1:-3]

    model = normalize_text(" ".join(model_tokens))
    if not brand or not model:
        return None
    canonical = f"{brand} {model}".strip()

    return SoundbarRecord(
        brand=brand,
        model=model,
        grade=grade,
        year=year,
        support_dolby_atmos=atmos,
        canonical=canonical,
    )


def load_soundbar_db_from_py(soundbar_list_py: Path) -> list[SoundbarRecord]:
    """
    `soundbar_list.py`에서 soundbar DB를 로드합니다.

    Args:
        soundbar_list_py: `soundbar_list.py` 파일 경로.

    Returns:
        SoundbarRecord 리스트(중복 canonical은 최초 1개 유지).

    Raises:
        FileNotFoundError: 파일이 없을 때.
        OSError: 파일 읽기 실패.
        ValueError: `data = [...]` 파싱 실패.
    """
    try:
        text = soundbar_list_py.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as e:
        raise OSError(f"사운드바 DB 파일 읽기 실패: {soundbar_list_py}") from e

    # soundbar_list.py에서 `data = [...]`를 안전하게 추출하기 위해 ast 사용
    try:
        module = ast.parse(text)
    except SyntaxError as e:
        raise ValueError(f"soundbar_list.py 파싱 실패: {soundbar_list_py}") from e

    data_value = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "data":
                    data_value = node.value
                    break
        if data_value is not None:
            break

    if data_value is None:
        raise ValueError("soundbar_list.py에서 `data` 변수를 찾지 못했습니다.")

    try:
        data_list = ast.literal_eval(data_value)
    except Exception as e:  # noqa: BLE001 - 안전한 오류 래핑
        raise ValueError("`data = [...]` 값을 literal_eval로 해석하지 못했습니다.") from e

    if not isinstance(data_list, list):
        raise ValueError("`data`는 list 타입이어야 합니다.")

    records: list[SoundbarRecord] = []
    canonicals: list[str] = []
    tmp: dict[str, SoundbarRecord] = {}
    for item in data_list:
        if not isinstance(item, str):
            continue
        rec = parse_soundbar_item(item)
        if rec is None:
            continue
        canonicals.append(rec.canonical)
        # 중복 canonical은 최초 등장 유지
        if rec.canonical not in tmp:
            tmp[rec.canonical] = rec

    ordered = unique_preserve_order(canonicals)
    for c in ordered:
        r = tmp.get(c)
        if r:
            records.append(r)
    return records


def get_brand_set(records: Iterable[SoundbarRecord]) -> set[str]:
    """
    레코드 리스트로부터 브랜드 집합을 구합니다.

    Args:
        records: SoundbarRecord iterable.

    Returns:
        브랜드 집합(정규화된 문자열).
    """
    return {r.brand for r in records if r.brand}

