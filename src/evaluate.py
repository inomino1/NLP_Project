"""
평가(evaluation) 모듈입니다.

지원 지표:
- Exact match Accuracy
- Micro/Macro F1 (UNKNOWN 포함/제외 옵션)
- Top-k Recall (retrieval 품질)
- MRR (랭킹 품질)
- threshold sweep: accept_threshold / margin_threshold 탐색

라벨 포맷: true_model 컬럼 필수. pred와 동일한 행 수·순서여야 합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .verify import UNKNOWN_LABEL


def _safe_import_sklearn_metrics():
    """scikit-learn metrics를 지연 로드합니다."""
    try:
        from sklearn.metrics import f1_score  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "scikit-learn이 필요합니다. `pip install scikit-learn`로 설치해 주세요."
        ) from e
    return f1_score


def json_loads_safe(text: Any, default: Any) -> Any:
    """
    JSON 문자열을 안전하게 파싱합니다.

    Args:
        text: JSON 문자열 또는 기타 타입
        default: 파싱 실패 시 반환값

    Returns:
        파싱된 객체 또는 default
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return default
    if isinstance(text, (list, dict)):
        return text
    try:
        return json.loads(str(text))
    except Exception:
        return default


@dataclass(frozen=True)
class EvalMetrics:
    """평가 지표 묶음입니다."""

    n: int
    accuracy: float
    f1_micro: float
    f1_macro: float
    top1_recall: float
    top3_recall: float
    top5_recall: float
    mrr: float


def _to_label(x: Optional[str]) -> str:
    """None을 UNKNOWN으로 통일합니다."""
    if x is None:
        return UNKNOWN_LABEL
    s = str(x).strip()
    return s if s else UNKNOWN_LABEL


def compute_topk_recall(y_true: list[str], topk: list[list[str]], k: int) -> float:
    """
    Top-k recall을 계산합니다.

    Args:
        y_true: 정답 라벨 리스트
        topk: 각 샘플의 ranked 후보 리스트(문자열)
        k: k

    Returns:
        Top-k recall
    """
    if not y_true:
        return 0.0
    hit = 0
    for t, cands in zip(y_true, topk, strict=False):
        if t in (cands or [])[:k]:
            hit += 1
    return float(hit) / float(len(y_true))


def compute_mrr(y_true: list[str], ranked: list[list[str]]) -> float:
    """
    Mean Reciprocal Rank(MRR)을 계산합니다.

    Args:
        y_true: 정답 라벨 리스트
        ranked: 각 샘플의 ranked 후보 리스트

    Returns:
        MRR
    """
    if not y_true:
        return 0.0
    rr_sum = 0.0
    for t, cands in zip(y_true, ranked, strict=False):
        if not cands:
            continue
        try:
            rank = cands.index(t) + 1
            rr_sum += 1.0 / float(rank)
        except ValueError:
            rr_sum += 0.0
    return rr_sum / float(len(y_true))


def evaluate_predictions(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    pred_col: str = "predicted_model",
    true_col: str = "true_model",
    top5_candidates_col: str = "top5_candidates",
    unknown_label: str = UNKNOWN_LABEL,
    include_unknown_in_f1: bool = True,
    exclude_both_unknown: bool = False,
) -> EvalMetrics:
    """
    예측 결과와 라벨을 첫 행부터 같은 행 위치끼리 매칭하여 지표를 계산합니다.

    pred와 labels의 행 수가 같고 순서가 동일하다는 전제입니다.

    Args:
        pred_df: `SoundbarModelAgent.batch_predict()` 결과
        labels_df: 라벨 DataFrame
        pred_col: 예측 컬럼명
        true_col: 정답 컬럼명
        top5_candidates_col: ranked 후보 컬럼명(JSON)
        unknown_label: UNKNOWN 라벨
        include_unknown_in_f1: UNKNOWN을 F1 계산에 포함할지 여부
        exclude_both_unknown: True면 예측·정답 둘 다 UNKNOWN인 행 제외 후 평가

    Returns:
        EvalMetrics
    """
    n_pred = len(pred_df)
    n_labels = len(labels_df)
    if n_pred != n_labels:
        raise ValueError(
            f"pred({n_pred}행)과 labels({n_labels}행)의 행 수가 같아야 합니다."
        )
    merged = pred_df.drop(columns=[true_col], errors="ignore").copy()
    merged[true_col] = labels_df[true_col].values
    merged[pred_col] = merged[pred_col].fillna(unknown_label).astype(str)
    merged[true_col] = merged[true_col].fillna(unknown_label).astype(str)

    y_true = [_to_label(x) for x in merged[true_col].tolist()]
    y_pred = [_to_label(x) for x in merged[pred_col].tolist()]

    # 예측·정답 둘 다 UNKNOWN인 행 제외
    if exclude_both_unknown:
        keep = [
            i
            for i, (t, p) in enumerate(zip(y_true, y_pred, strict=False))
            if not (t == unknown_label and p == unknown_label)
        ]
        y_true = [y_true[i] for i in keep]
        y_pred = [y_pred[i] for i in keep]
        merged = merged.iloc[keep].reset_index(drop=True)

    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=False)])) if y_true else 0.0

    f1_score = _safe_import_sklearn_metrics()
    if include_unknown_in_f1:
        labels = sorted(set(y_true) | set(y_pred))
    else:
        labels = sorted({x for x in (set(y_true) | set(y_pred)) if x != unknown_label})

    f1_micro = float(f1_score(y_true, y_pred, labels=labels, average="micro")) if labels else 0.0
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro")) if labels else 0.0

    # ranked 후보 추출
    ranked: list[list[str]] = []
    for v in merged.get(top5_candidates_col, pd.Series([None] * len(merged))).tolist():
        items = json_loads_safe(v, default=[])
        if isinstance(items, list):
            ranked.append([str(x.get("canonical")) for x in items if isinstance(x, dict) and x.get("canonical")])
        else:
            ranked.append([])

    top1 = compute_topk_recall(y_true, ranked, 1)
    top3 = compute_topk_recall(y_true, ranked, 3)
    top5 = compute_topk_recall(y_true, ranked, 5)
    mrr = compute_mrr(y_true, ranked)

    return EvalMetrics(
        n=len(y_true),
        accuracy=accuracy,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        top1_recall=top1,
        top3_recall=top3,
        top5_recall=top5,
        mrr=mrr,
    )


def sweep_thresholds(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    true_col: str = "true_model",
    top5_candidates_col: str = "top5_candidates",
    unknown_label: str = UNKNOWN_LABEL,
    accept_thresholds: Optional[list[float]] = None,
    margin_thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    간단한 규칙(Top1 score / margin) 기반으로 threshold를 탐색합니다.

    규칙:
    - top1_score >= accept_threshold AND (top1_score - top2_score) >= margin_threshold 이면 top1 채택
    - 아니면 UNKNOWN

    pred와 labels는 첫 행부터 같은 행 위치끼리 매칭됩니다.

    Args:
        pred_df: `batch_predict` 결과(최소 top5_candidates 필요)
        labels_df: 라벨 DF
        accept_thresholds: accept 후보값 리스트(기본: 0.1~0.95)
        margin_thresholds: margin 후보값 리스트(기본: 0.0~0.2)

    Returns:
        threshold 조합별 metric DataFrame (macro_f1 기준으로 비교 권장)
    """
    if accept_thresholds is None:
        accept_thresholds = [round(x, 2) for x in np.linspace(0.1, 0.95, 18).tolist()]
    if margin_thresholds is None:
        margin_thresholds = [round(x, 3) for x in np.linspace(0.0, 0.2, 11).tolist()]

    n_pred = len(pred_df)
    n_labels = len(labels_df)
    if n_pred != n_labels:
        raise ValueError(
            f"pred({n_pred}행)과 labels({n_labels}행)의 행 수가 같아야 합니다."
        )
    merged = pred_df.drop(columns=[true_col], errors="ignore").copy()
    merged[true_col] = labels_df[true_col].values
    y_true = [_to_label(x) for x in merged[true_col].fillna(unknown_label).astype(str).tolist()]

    # 후보/점수 추출
    top1_can: list[str] = []
    top1_score: list[float] = []
    top2_score: list[float] = []
    for v in merged.get(top5_candidates_col, pd.Series([None] * len(merged))).tolist():
        items = json_loads_safe(v, default=[])
        if not isinstance(items, list) or not items:
            top1_can.append(unknown_label)
            top1_score.append(float("-inf"))
            top2_score.append(float("-inf"))
            continue
        c1 = items[0] if isinstance(items[0], dict) else None
        c2 = items[1] if len(items) > 1 and isinstance(items[1], dict) else None
        top1_can.append(str(c1.get("canonical")) if c1 and c1.get("canonical") else unknown_label)
        top1_score.append(float(c1.get("score")) if c1 and c1.get("score") is not None else float("-inf"))
        top2_score.append(float(c2.get("score")) if c2 and c2.get("score") is not None else float("-inf"))

    f1_score = _safe_import_sklearn_metrics()
    labels = sorted(set(y_true) | {unknown_label} | set(top1_can))

    rows: list[dict[str, Any]] = []
    for a in accept_thresholds:
        for m in margin_thresholds:
            y_pred = []
            for can, s1, s2 in zip(top1_can, top1_score, top2_score, strict=False):
                if s1 >= a and (s1 - s2) >= m:
                    y_pred.append(can)
                else:
                    y_pred.append(unknown_label)
            acc = float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=False)])) if y_true else 0.0
            f1_micro = float(f1_score(y_true, y_pred, labels=labels, average="micro")) if labels else 0.0
            f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro")) if labels else 0.0
            rows.append(
                {
                    "accept_threshold": a,
                    "margin_threshold": m,
                    "accuracy": acc,
                    "f1_micro": f1_micro,
                    "f1_macro": f1_macro,
                    "n": len(y_true),
                }
            )
    return pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)

