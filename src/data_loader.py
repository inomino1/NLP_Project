"""
CSV 입력 로드(파일 I/O) 유틸리티 모듈입니다.

- `raw_data/HDMI_BT_Log.csv` 로드
- 라벨 CSV 로드 (평가용)

모든 파일 입출력에는 오류 처리가 포함됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CsvLoadResult:
    """CSV 로드 결과(데이터+메타)를 담습니다."""

    path: Path
    df: pd.DataFrame


def load_hdmi_bt_log_csv(path: Path, encoding: str = "utf-8") -> CsvLoadResult:
    """
    HDMI_BT_Log.csv 포맷의 로그 CSV를 로드합니다 (기본 경로: raw_data/).

    Args:
        path: CSV 경로.
        encoding: 파일 인코딩(기본 utf-8). 필요 시 `cp949` 등으로 변경.

    Returns:
        CsvLoadResult

    Raises:
        FileNotFoundError: 파일이 없을 때.
        OSError: 파일 읽기 실패.
        ValueError: CSV 파싱 실패 또는 필수 컬럼 부재.
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        raise
    except UnicodeDecodeError as e:
        raise OSError(f"CSV 인코딩 오류: {path} (encoding={encoding})") from e
    except Exception as e:  # noqa: BLE001
        raise OSError(f"CSV 로드 실패: {path}") from e

    required_cols = [
        "NAME_BT",
        "NAME1",
        "BRAND1",
        "NAME2",
        "BRAND2",
        "NAME3",
        "BRAND3",
        "NAME4",
        "BRAND4",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    return CsvLoadResult(path=path, df=df)


def load_labels_csv(
    path: Path,
    encoding: str = "utf-8",
    true_model_col: str = "true_model",
) -> CsvLoadResult:
    """
    수작업 라벨 CSV를 로드합니다.

    라벨 포맷: `true_model` 컬럼 필수 (정답 canonical 또는 "UNKNOWN").
    pred와 동일한 행 수·순서여야 합니다.

    Args:
        path: 라벨 CSV 경로.
        encoding: 인코딩.
        true_model_col: 정답 모델 컬럼명.

    Returns:
        CsvLoadResult

    Raises:
        FileNotFoundError: 파일이 없을 때.
        OSError: 파일 읽기 실패.
        ValueError: 컬럼 부재 또는 타입 문제.
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        raise
    except UnicodeDecodeError as e:
        raise OSError(f"라벨 CSV 인코딩 오류: {path} (encoding={encoding})") from e
    except Exception as e:  # noqa: BLE001
        raise OSError(f"라벨 CSV 로드 실패: {path}") from e

    if true_model_col not in df.columns:
        raise ValueError(
            f"라벨 CSV는 `{true_model_col}` 컬럼을 포함해야 합니다."
        )

    df[true_model_col] = df[true_model_col].astype(str)
    return CsvLoadResult(path=path, df=df)

