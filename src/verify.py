"""
UNKNOWN 결정 규칙 및 (옵션) 후보 제한 LLM 검증(verifier) 모듈입니다.

목표:
- 검색(retrieval) 결과가 불확실할 때, \"틀릴 바엔 UNKNOWN\"을 선택할 수 있도록 함
- 점수 threshold, 1-2위 margin, 입력 노이즈(BT 헤드폰 등) 신호를 반영
- (옵션) 상위 후보 목록 안에서만 선택하는 closed-book LLM verifier 제공
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Set

from .brand_extractor import extract_brand_from_query
from .device_type import is_non_soundbar_device
from .log_features import LogQuery, bt_looks_like_non_soundbar, choose_primary_query
from .retrieval import RetrievedCandidate

# 질의 첫 토큰이 아래이면 generic 입력소스(TV IN, AV 등)로 간주. 알려진 브랜드 없을 때 UNKNOWN.
_GENERIC_INPUT_FIRST_TOKENS: frozenset[str] = frozenset(
    {"TV", "IN", "AV", "HDMI", "USB", "OPTICAL", "ARC", "AUX", "DIGITAL"}
)

UNKNOWN_LABEL = "UNKNOWN"


def _get_canonical_model_part(canonical: str) -> str:
    """canonical 'BRAND MODEL'에서 모델 부분만 추출합니다."""
    parts = canonical.upper().split()
    return " ".join(parts[1:]) if len(parts) > 1 else ""


def _compact_no_space(s: str) -> str:
    """공백 제거 후 문자열. ARC ULTRA vs ARCULTRA 매칭용."""
    return "".join(s.upper().split())


def _has_lexical_overlap(query: str, canonical: str, min_token_len: int = 2) -> bool:
    """
    질의와 canonical 간 의미 있는 어휘 중첩이 있는지 검사합니다.

    기기명 블랙리스트 없이, 질의가 사운드바 DB 후보와 무관할 때 UNKNOWN을
    선택하는 데 사용합니다. (예: "SKY Q" vs "SONY HTS40R" -> 중첩 없음 -> UNKNOWN)
    ARC ULTRA vs ARCULTRA: 공백 제거 후 동일하면 중첩으로 인정합니다.

    Args:
        query: 정규화된 질의 문자열
        canonical: "BRAND MODEL" 형식의 canonical 또는 model_part
        min_token_len: 중첩 판별 시 최소 토큰 길이(단일 문자 제외)

    Returns:
        중첩 있으면 True, 없으면 False
    """
    # 공백 제거 정규화: ARC ULTRA == ARCULTRA
    q_compact = _compact_no_space(query)
    c_compact = _compact_no_space(canonical)
    if q_compact and c_compact and (q_compact in c_compact or c_compact in q_compact):
        return True

    q_tokens = [t for t in query.upper().split() if len(t) >= min_token_len]
    c_tokens = [t for t in canonical.upper().split() if len(t) >= min_token_len]
    if not q_tokens or not c_tokens:
        return False
    for q in q_tokens:
        for c in c_tokens:
            if q in c or c in q:
                return True
    return False


def _levenshtein_distance(a: str, b: str) -> int:
    """
    Levenshtein distance(편집 거리)를 계산합니다.
    오타/유사 모델 overlap 판별에 사용합니다.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    cur[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def _has_typo_overlap(query_part: str, canon_part: str, max_edit_distance: int = 2) -> bool:
    """
    오타/유사 모델 overlap을 검사합니다.
    LG US20A vs LG S20A, LG NS60TR vs LG S60TR 등 1~2자 차이 허용.

    Args:
        query_part: 질의의 모델 부분
        canon_part: canonical의 모델 부분
        max_edit_distance: 허용 최대 편집 거리

    Returns:
        오타 수준 overlap이면 True
    """
    q = (query_part or "").upper().replace(" ", "")
    c = (canon_part or "").upper().replace(" ", "")
    if not q or not c:
        return False
    # 3자 이상 공통 서브스트링
    for ln in range(3, min(len(q), len(c)) + 1):
        for i in range(len(q) - ln + 1):
            sub = q[i : i + ln]
            if sub in c:
                return True
    # Levenshtein 거리 <= 2 (길이 비슷할 때)
    if abs(len(q) - len(c)) <= 2 and _levenshtein_distance(q, c) <= max_edit_distance:
        return True
    return False


def _extract_model_numbers(text: str) -> list[int]:
    """모델명에서 숫자 시퀀스를 추출합니다. ENCHANT900→[900], S20A→[20]."""
    parts = re.findall(r"\d+", (text or "").upper())
    return [int(p) for p in parts if p]


def _pick_closest_model_candidate(
    query: str, candidates: list[RetrievedCandidate]
) -> Optional[RetrievedCandidate]:
    """
    small_margin 시 쿼리와 모델 번호가 가장 가까운 후보를 선택합니다.
    ENCHANT900 → ENCHANT800(800) 선호, ENCHANT1300(1300) 비선호.
    """
    q_nums = _extract_model_numbers(query)
    if not q_nums:
        return candidates[0] if candidates else None
    q_num = q_nums[-1]  # 주로 마지막 숫자가 모델 번호(900, 800 등)
    best: Optional[tuple[RetrievedCandidate, int]] = None
    for c in candidates:
        canon_part = _get_canonical_model_part(c.canonical) or c.canonical
        c_nums = _extract_model_numbers(canon_part)
        if not c_nums:
            continue
        c_num = c_nums[-1]
        diff = abs(q_num - c_num)
        if best is None or diff < best[1]:
            best = (c, diff)
    return best[0] if best else (candidates[0] if candidates else None)


def _has_model_part_overlap(query: str, canonical: str, min_substring_len: int = 4) -> bool:
    """
    질의와 canonical의 모델 부분 간 거의 일치 수준 중첩이 있는지 검사합니다.

    브랜드를 참조하지 못할 때, 모델명이 거의 일치해야만 예측하도록 합니다.
    (예: "SKY Q" vs "SONOS ARC" → model "ARC"와 중첩 없음 → UNKNOWN)
    (예: "CINEBAR 11" vs "JBL CINEMASB110" → "CINE" 등 4자 이상 중첩 → accept)

    Args:
        query: 정규화된 질의 문자열
        canonical: "BRAND MODEL" 형식의 canonical
        min_substring_len: 실질적 일치로 인정할 최소 중첩 길이

    Returns:
        모델 부분과 실질적 중첩 있으면 True
    """
    parts = canonical.upper().split()
    model_part = " ".join(parts[1:]) if len(parts) > 1 else ""
    if not model_part:
        return False
    q_tokens = [t for t in query.upper().split() if len(t) >= min_substring_len]
    if not q_tokens:
        return False
    for q in q_tokens:
        for i in range(len(q) - min_substring_len + 1):
            sub = q[i : i + min_substring_len]
            if sub in model_part:
                return True
        if q in model_part or model_part in q:
            return True
    return False


@dataclass(frozen=True)
class VerificationConfig:
    """검증/결정 로직 설정입니다."""

    accept_score_threshold: float = 0.52  # 0.55→0.52 (LG NS60TR 등 BT 가중치 완화)
    margin_threshold: float = 0.05
    prefer_unknown_on_uncertain: bool = True
    allow_bt_only_prediction: bool = False
    unknown_label: str = UNKNOWN_LABEL
    known_brands: Optional[Set[str]] = None  # 사운드바 DB 브랜드 집합, NER 판별용


@dataclass(frozen=True)
class Prediction:
    """
    최종 예측 결과입니다.

    Attributes:
        canonical_model: 예측된 canonical 모델명. UNKNOWN이면 None.
        confidence: [0, 1] 범위의 대략적 신뢰도(휴리스틱).
        evidence: 근거(입력 질의, 상위 후보/점수 등).
    """

    canonical_model: Optional[str]
    confidence: float
    evidence: dict[str, Any]


class RuleBasedVerifier:
    """
    retrieval 결과를 기반으로 UNKNOWN/모델을 결정하는 규칙 기반 verifier 입니다.

    Public API:
    - `verify(queries, candidates) -> Prediction`
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_verifier: Optional["ClosedBookLLMVerifier"] = None,
    ) -> None:
        """
        Args:
            config: 검증 설정
            llm_verifier: (옵션) closed-book LLM verifier
        """
        self.config = config or VerificationConfig()
        self.llm_verifier = llm_verifier

    def verify(self, queries: list[LogQuery], candidates: list[RetrievedCandidate]) -> Prediction:
        """
        후보 리스트에서 최종 모델 또는 UNKNOWN을 선택합니다.

        Args:
            queries: 로그에서 추출된 질의 리스트
            candidates: retrieval 결과 (score 내림차순 권장)

        Returns:
            Prediction
        """
        primary = choose_primary_query(queries)
        if primary is None:
            return Prediction(
                canonical_model=None,
                confidence=1.0,
                evidence={"reason": "no_valid_query"},
            )

        if not candidates:
            return Prediction(
                canonical_model=None,
                confidence=1.0,
                evidence={"reason": "no_candidates", "primary_query": primary.query},
            )

        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None
        margin = (top1.score - top2.score) if top2 is not None else 1.0

        # 룰: 브랜드 없이 "SOUNDBAR" 또는 "SOUND BAR" 단독 → ETC SOUNDBAR
        q_compact = _compact_no_space(primary.query)
        if q_compact == "SOUNDBAR":
            return Prediction(
                canonical_model="ETC SOUNDBAR",
                confidence=1.0,
                evidence={
                    "reason": "generic_soundbar_rule",
                    "primary_query": primary.query,
                },
            )

        # Entity-based: display primary(queries[0])이 비사운드바 기기면 UNKNOWN
        # APPLE TV, SKY Q, ORANGE 디코더 등 입력 소스는 사운드바가 아님
        if queries and is_non_soundbar_device(queries[0].query):
            return Prediction(
                canonical_model=None,
                confidence=0.85,
                evidence={
                    "reason": "non_soundbar_device",
                    "primary_query": queries[0].query,
                    "top1": top1.__dict__,
                },
            )

        # NER 기반: 질의에서 브랜드 추출 (known_brands = 사운드바 DB 브랜드 집합)
        known_brands = self.config.known_brands or set()
        extraction = (
            extract_brand_from_query(primary.query, known_brands) if known_brands else None
        )
        query_has_known_brand = extraction is not None and extraction.brand is not None

        # 브랜드 없음: no_known_brand로 UNKNOWN 반환하지 않음.
        # 모델명 + 사운드바 DB 간 코사인 유사도로 매칭(agent의 cosine gate에서 처리, 공백 제거 적용)

        soundbar_hint_re = re.compile(r"\b(SOUNDBAR|SOUND\s*BAR)\b")
        # overlap 검사 제거: cosine 유사도(0.70~0.85)로만 accept/reject 결정 (agent cosine gate)

        # "브랜드 + SOUNDBAR"처럼 모델이 부정확한 입력은, 브랜드의 generic soundbar 엔트리를 우대
        # - 예: "LG SOUND BAR" -> "LG LGSOUNDBAR" (DB에 존재하는 경우)
        generic_soundbar_relax_threshold = 0.45
        if soundbar_hint_re.search(primary.query):
            # query의 첫 토큰을 브랜드로 가정 (예: "LG LG SOUND BAR"의 첫 토큰 "LG")
            parts = primary.query.split()
            brand = parts[0] if parts else ""
            if brand and top1.canonical.startswith(f"{brand} "):
                # canonical 안에 "SOUNDBAR"가 포함된 generic 모델이면 낮은 threshold로 accept
                if "SOUNDBAR" in top1.canonical and top1.score >= generic_soundbar_relax_threshold:
                    return Prediction(
                        canonical_model=top1.canonical,
                        confidence=float(min(1.0, max(0.0, top1.score + 0.25))),
                        evidence={
                            "reason": "generic_soundbar_accept",
                            "primary_query": primary.query,
                            "top1": top1.__dict__,
                        },
                    )

        # BT만 있는 경우(헤드폰/이어폰 가능성이 높음) 보수적으로 UNKNOWN
        if (
            primary.source == "NAME_BT"
            and bt_looks_like_non_soundbar(primary.query)
            and not self.config.allow_bt_only_prediction
        ):
            # 매우 높은 점수면 예외 허용
            if top1.score < max(self.config.accept_score_threshold, 0.75):
                return Prediction(
                    canonical_model=None,
                    confidence=0.9,
                    evidence={
                        "reason": "bt_noise",
                        "primary_query": primary.query,
                        "top1": top1.__dict__,
                    },
                )

        # threshold 미만이면 UNKNOWN
        if top1.score < self.config.accept_score_threshold:
            return Prediction(
                canonical_model=None,
                confidence=1.0 - min(1.0, top1.score),
                evidence={
                    "reason": "below_threshold",
                    "primary_query": primary.query,
                    "top1": top1.__dict__,
                    "threshold": self.config.accept_score_threshold,
                },
            )

        # margin이 작으면 불확실 -> (옵션) LLM 검증
        # overlap 기반 margin 완화 제거, cosine gate에서 처리
        margin_thresh = self.config.margin_threshold
        if top2 is not None and margin < margin_thresh:
            if self.llm_verifier is not None:
                selected = self.llm_verifier.select(primary.query, candidates[:5])
                if selected is None:
                    return Prediction(
                        canonical_model=None,
                        confidence=0.6,
                        evidence={
                            "reason": "llm_unknown",
                            "primary_query": primary.query,
                            "top5": [c.__dict__ for c in candidates[:5]],
                        },
                    )
                return Prediction(
                    canonical_model=selected,
                    confidence=0.65,
                    evidence={
                        "reason": "llm_selected",
                        "primary_query": primary.query,
                        "selected": selected,
                        "top5": [c.__dict__ for c in candidates[:5]],
                    },
                )

            if self.config.prefer_unknown_on_uncertain:
                # cosine 기반: 모델 번호가 쿼리와 가장 가까운 후보 선택 (ENCHANT900→ENCHANT800 등)
                closest = _pick_closest_model_candidate(primary.query, candidates[:5])
                if closest is not None:
                    return Prediction(
                        canonical_model=closest.canonical,
                        confidence=0.6,
                        evidence={
                            "reason": "small_margin_closest_model",
                            "primary_query": primary.query,
                            "selected": closest.__dict__,
                            "margin": margin,
                            "margin_threshold": self.config.margin_threshold,
                        },
                    )
                return Prediction(
                    canonical_model=None,
                    confidence=0.55,
                    evidence={
                        "reason": "small_margin",
                        "primary_query": primary.query,
                        "top1": top1.__dict__,
                        "top2": top2.__dict__,
                        "margin": margin,
                        "margin_threshold": self.config.margin_threshold,
                    },
                )

        # 기본: top1 채택
        confidence = float(min(1.0, max(0.0, top1.score)))
        return Prediction(
            canonical_model=top1.canonical,
            confidence=confidence,
            evidence={
                "reason": "accepted",
                "primary_query": primary.query,
                "top1": top1.__dict__,
                "margin": margin,
            },
        )


class ClosedBookLLMVerifier:
    """
    후보 목록(topN) 안에서만 답하도록 강제하는 LLM verifier 입니다.

    이 구현은 transformers 기반 로컬 모델을 가정하며, 미설치/미로드 환경에서는
    ImportError/RuntimeError를 발생시킵니다.

    Public API:
    - `select(observed, candidates) -> canonical|None`
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model_id: HuggingFace 모델 ID (예: "Qwen/Qwen2.5-1.5B-Instruct")
            device: "auto" 또는 "cpu"/"cuda" 등
            max_new_tokens: 생성 토큰 수
            temperature: 샘플링 온도(0이면 거의 결정적)
        """
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._generator = self._init_generator()

    def _init_generator(self):
        """transformers text-generation 파이프라인을 초기화합니다."""
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "transformers가 필요합니다. `pip install transformers`로 설치해 주세요."
            ) from e

        # device="auto" 지원 여부는 버전에 따라 다를 수 있어 예외를 허용
        try:
            return pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device,
            )
        except Exception:
            # fallback: device_map 없이 시도
            return pipeline("text-generation", model=self.model_id)

    def select(
        self, observed: str, candidates: list[RetrievedCandidate], unknown_label: str = UNKNOWN_LABEL
    ) -> Optional[str]:
        """
        관측 문자열과 후보 목록을 입력으로 받아, 후보 중 하나 또는 UNKNOWN을 선택합니다.

        Args:
            observed: 관측된 기기명/질의 문자열
            candidates: 상위 후보 리스트(보통 Top5)
            unknown_label: UNKNOWN 라벨 문자열

        Returns:
            선택된 canonical 문자열 또는 None(UNKNOWN)
        """
        if not candidates:
            return None

        # closed-book 강제를 위해 후보 목록을 명시하고, JSON으로만 답하게 지시
        options = [c.canonical for c in candidates]
        prompt = self._build_prompt(observed, options, unknown_label=unknown_label)
        out = self._generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=self.temperature,
        )
        text = self._extract_generated_text(out)
        selected = self._parse_selection(text, options, unknown_label=unknown_label)
        return selected

    def _build_prompt(self, observed: str, options: list[str], unknown_label: str) -> str:
        """LLM 프롬프트를 생성합니다."""
        payload = {
            "observed": observed,
            "options": options,
            "rules": [
                "정답은 options 중 하나를 그대로 선택하거나 UNKNOWN을 선택한다.",
                "options에 없는 문자열을 생성하지 않는다.",
                "출력은 JSON 한 줄만 반환한다: {\"choice\": \"...\", \"reason\": \"...\"}",
            ],
        }
        return (
            "다음은 사운드바 모델 매핑 문제이다.\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "JSON으로만 답해라.\n"
        )

    def _extract_generated_text(self, out: Any) -> str:
        """transformers pipeline 출력에서 생성 텍스트를 추출합니다."""
        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict) and "generated_text" in item:
                return str(item["generated_text"])
        return str(out)

    def _parse_selection(
        self, generated_text: str, options: list[str], unknown_label: str
    ) -> Optional[str]:
        """
        생성 결과에서 choice를 파싱합니다.

        JSON 파싱 실패 시, options에 대한 문자열 포함 여부로 보조 추정합니다.
        """
        # 가장 마지막 JSON 객체를 찾기 위해 간단히 {...} 블록을 스캔
        m = re.findall(r"\{[\s\S]*?\}", generated_text)
        for blob in reversed(m):
            try:
                obj = json.loads(blob)
                choice = str(obj.get("choice", "")).strip()
                if not choice or choice.upper() == unknown_label:
                    return None
                if choice in options:
                    return choice
            except Exception:
                continue

        # fallback: 옵션 문자열이 그대로 포함되어 있으면 그 중 첫 매칭 반환
        for opt in options:
            if opt in generated_text:
                return opt
        return None

