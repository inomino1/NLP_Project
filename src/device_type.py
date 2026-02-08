"""
Entity 기반 비사운드바 기기(스트리밍/셋톱박스/디코더 등) 탐지 모듈입니다.

블랙리스트 방식이 아닌, 질의가 "사운드바가 아닌 입력 소스"를 지칭하는지
패턴/엔티티 기반으로 판별합니다. 이를 통해 APPLE TV, SKY Q, ORANGE 디코더 등
입력 소스명이 사운드바로 잘못 매칭되는 것을 방지합니다.
"""

from __future__ import annotations

import re


# 스트리밍/미디어 플레이어 (사운드바 브랜드 아님)
_STREAMING_PATTERNS: tuple[str, ...] = (
    r"\bAPPLE\s+TV\b",
    r"\bFIRE\s+TV\b",
    r"\bAMAZON\s+FIRE\s+T\b",
    r"\bFIRETV\b",
    r"\bCHROMECAST\b",
    r"\bROKU\b",
    r"\bMAGENTATV\b",
    r"\bTELIA\s+TV\b",
    r"\bVODAFONE\s+TV\b",
    r"\bEE\s+TV\b",  # EE TV (BT/영국 통신사 스트리밍)
    r"\bSHIELD\b",  # NVIDIA Shield 셋톱박스
)

# 셋톱박스/위성 수신기
_SETTOP_PATTERNS: tuple[str, ...] = (
    r"\bSKY\b",  # SKY 단독 (SKY Q 외에 SKY, SKY+ 등)
    r"\bSKY\s+Q\b",
    r"\bSETTOP\s+BOX\b",
    r"\bSET\s+TOP\s+BOX\b",
    r"\bSTB\b",
    r"\bSF8008\b",  # 삼성/기타 셋톱박스 모델
    r"\bMEDIABOX\b",  # 미디어박스 (MEDIABOX 1 등)
)

# 게임 콘솔 (단독 질의로 사용될 때)
_CONSOLE_PATTERNS: tuple[str, ...] = (
    r"\bXBOX\b",
    r"\bPLAYSTATION\s+\d+\b",  # PLAYSTATION 4, 5, 6 등 모든 번호
    r"\bPS\d+\b",  # PS4, PS5, PS6 등 모든 번호
    r"\bNINTENDO\b",  # NINTENDO SWITCH, WII, WII U, DS, 3DS 등 모든 콘솔
    r"\bCONSOLA\s+DE\s+JUEGOS\b",
)

# 일반 미디어 소스 (사운드바 아님) - 디코더/인터넷 박스 포함
_GENERIC_SOURCE_PATTERNS: tuple[str, ...] = (
    r"\bBLURAY\s+PLAYER\b",
    r"\bBD\s+PLAYER\b",
    r"\bDVD\b",
    r"\bDVDPLAYER\b",
    r"\bAV\s+RECEIVER\b",
    r"\bTV\s+IN\b",  # TV IN WOONKAME 등: TV 입력 소스 라벨, 사운드바 아님
    # 디코더/인터넷 박스 (다국어) - 통신사(ORANGE 등) 조합은 cosine 유사도로 처리
    r"\bCODEUR\b",
    r"\bDECODER\b",
    r"\bBOX\s+INTERNET\b",
    r"\bBO\s+TIER\s+D\b",
    r"\bFIBRE\s+OPTIQUE\b",
)

# 사운드바 DB 브랜드인 경우 무시 (예: SONY BARRA DE SOM)
_KNOWN_SOUNDBAR_BRAND_PREFIX = (
    "LG ",
    "SAMSUNG ",
    "SONY ",
    "BOSE ",
    "JBL ",
    "PHILIPS ",
    "DENON ",
    "SONOS ",
    "TCL ",
    "HISENSE ",
    "HARMAN ",
    "YAMAHA ",
    "VIZIO ",
    "POLK ",
    "KLIPSCH ",
    "ROKU ",
)

_streaming_re = re.compile("|".join(_STREAMING_PATTERNS), re.IGNORECASE)
_settop_re = re.compile("|".join(_SETTOP_PATTERNS), re.IGNORECASE)
_console_re = re.compile("|".join(_CONSOLE_PATTERNS), re.IGNORECASE)
_generic_re = re.compile("|".join(_GENERIC_SOURCE_PATTERNS), re.IGNORECASE)


def is_non_soundbar_device(query: str) -> bool:
    """
    질의가 사운드바가 아닌 입력 소스(스트리밍/셋톱박스/디코더 등)를 지칭하는지 판별합니다.

    블랙리스트가 아닌 패턴 기반입니다. 예:
    - "APPLE TV" -> True (스트리밍 기기)
    - "SKY Q" -> True (셋톱박스)
    - "ORANGE FIBRE OPTIQUE BO TIER D CODEUR /BOX INTERNET" -> True (디코더)
    - "LG S40T" -> False (사운드바)
    - "SONY SONY BARRA DE SOM" -> False (사운드바)

    Args:
        query: 정규화된 질의 문자열

    Returns:
        비사운드바 기기이면 True, 사운드바일 가능성이 있으면 False
    """
    if not query or not query.strip():
        return False

    q = query.upper().strip()

    # 사운드바 DB 브랜드로 시작하면 비사운드바 아님 (예: LG S40T, SONY BARRA DE SOM)
    for prefix in _KNOWN_SOUNDBAR_BRAND_PREFIX:
        if q.startswith(prefix):
            return False

    # 스트리밍/셋톱/콘솔/일반 미디어(디코더 포함) 패턴 매칭
    if _streaming_re.search(q):
        return True
    if _settop_re.search(q):
        return True
    if _console_re.search(q):
        return True
    if _generic_re.search(q):
        return True

    return False
