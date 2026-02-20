# -*- coding: utf-8 -*-
"""
soundbar_model_agent.ipynb 를 Databricks용 단일 파일(모듈 인라인)로 생성합니다.
실행: python build_databricks_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def read_src(name: str) -> str:
    """src/{name}.py 내용 반환 (상대 import 제거)."""
    p = Path(__file__).resolve().parent / "src" / f"{name}.py"
    text = p.read_text(encoding="utf-8")
    # from .xxx import ... 제거 (노트북에서 같은 네임스페이스)
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("from .") or line.strip().startswith("import ."):
            continue
        lines.append(line)
    return "\n".join(lines)


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split("\n") if isinstance(source, str) else source}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n") if isinstance(source, str) else [source]}


def main() -> None:
    root = Path(__file__).resolve().parent
    cells = []

    # --- Title ---
    cells.append(md_cell("""# 사운드바 모델 매핑 Agent (Databricks용)

로그 CSV의 관측 기기명(NAME1~4, NAME_BT)을 표준 canonical 사운드바 모델명으로 매핑합니다.
**모든 로직이 이 노트북 안에 포함되어 있어, Databricks에 노트북만 올려서 실행할 수 있습니다.**

1. 경로 설정
2. 공통 모듈 코드 (정규화, DB, 검색, 검증, Agent)
3. 데이터 로드 및 예측
4. 결과 저장"""))

    # --- 1. 경로 설정 ---
    cells.append(md_cell("## 1. 경로 설정\n\nDatabricks에서는 DBFS 경로(예: `/dbfs/FileStore/...`) 또는 위젯으로 경로를 지정할 수 있습니다."))
    path_code = '''
from pathlib import Path

# 아래 경로를 환경에 맞게 수정하세요. Databricks: /dbfs/FileStore/... 등
INPUT_CSV = Path("/path/to/HDMI_BT_Log.csv")
SOUNDBAR_DB = Path("/path/to/soundbar_list.py")
OUTPUT_PRED = Path("/path/to/output/pred.csv")

OUTPUT_PRED.parent.mkdir(parents=True, exist_ok=True)
print("INPUT_CSV:", INPUT_CSV, "존재:", INPUT_CSV.exists())
print("SOUNDBAR_DB:", SOUNDBAR_DB, "존재:", SOUNDBAR_DB.exists())
'''
    cells.append(code_cell(path_code.strip()))

    # --- 2. 인라인 모듈 (순서 중요) ---
    cells.append(md_cell("## 2. 공통 모듈 코드 (정규화, 사운드바 DB, 로그 질의, 브랜드 추출, 비사운드바 탐지, 임베딩 검색, 검증, Agent)\n\n아래 셀들을 순서대로 실행하면 됩니다."))

    for name, title in [
        ("normalize", "정규화 (normalize)"),
        ("soundbar_db", "사운드바 DB (soundbar_db)"),
        ("log_features", "로그 질의 (log_features)"),
        ("brand_extractor", "브랜드 추출 (brand_extractor)"),
        ("device_type", "비사운드바 기기 탐지 (device_type)"),
        ("embedding_retriever", "임베딩 검색 (embedding_retriever)"),
        ("retrieval", "하이브리드 검색 (retrieval)"),
        ("verify", "검증 규칙 (verify)"),
    ]:
        cells.append(md_cell(f"### {title}"))
        cells.append(code_cell(read_src(name)))

    # data_loader: load_hdmi_bt_log_csv 만 (CsvLoadResult 포함)
    data_loader_src = '''
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class CsvLoadResult:
    path: Path
    df: pd.DataFrame

def load_hdmi_bt_log_csv(path: Path, encoding: str = "utf-8") -> CsvLoadResult:
    try:
        df = pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        raise
    except UnicodeDecodeError as e:
        raise OSError(f"CSV 인코딩 오류: {path}") from e
    except Exception as e:
        raise OSError(f"CSV 로드 실패: {path}") from e
    required_cols = ["NAME_BT", "NAME1", "BRAND1", "NAME2", "BRAND2", "NAME3", "BRAND3", "NAME4", "BRAND4"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    return CsvLoadResult(path=path, df=df)
'''
    cells.append(md_cell("### 데이터 로더 (load_hdmi_bt_log_csv)"))
    cells.append(code_cell(data_loader_src.strip()))

    # agent
    cells.append(md_cell("### Agent (SoundbarModelAgent)"))
    cells.append(code_cell(read_src("agent")))

    # --- 3. 데이터 로드 및 예측 ---
    cells.append(md_cell("## 3. 데이터 로드 및 예측"))
    load_predict_code = '''
import time
import pandas as pd

# CSV 로드
log_res = load_hdmi_bt_log_csv(INPUT_CSV, encoding="utf-8")
df_log = log_res.df
records = load_soundbar_db_from_py(SOUNDBAR_DB)
brand_set = get_brand_set(records)
print("로그 행 수:", len(df_log))
print("사운드바 DB 레코드 수:", len(records))

# Agent 생성 및 일괄 예측
config = AgentConfig(
    top_n_lexical=200,
    top_k=20,
    embedding_model_id="BAAI/bge-small-en-v1.5",
    use_embeddings=True,
    include_bt=True,
    accept_score_threshold=0.52,
    margin_threshold=0.05,
    accuracy_mode=True,
    min_cosine_similarity=0.85,
)
agent = SoundbarModelAgent(SOUNDBAR_DB, config=config)

t0 = time.perf_counter()
pred_df = agent.batch_predict(df_log)
elapsed = time.perf_counter() - t0
print(f"예측 완료: {len(pred_df)}행, 소요 시간: {elapsed:.1f}초")
display(pred_df.head(10))
'''
    cells.append(code_cell(load_predict_code.strip()))

    # --- 4. 결과 저장 ---
    cells.append(md_cell("## 4. 결과 저장"))
    save_code = '''
try:
    pred_df.to_csv(OUTPUT_PRED, index=False, encoding="utf-8")
    print("저장 완료:", OUTPUT_PRED)
except OSError as e:
    print("저장 실패:", e)
'''
    cells.append(code_cell(save_code.strip()))

    # Notebook JSON: source는 줄 단위 리스트, 각 줄 끝에 \n
    for c in cells:
        if c["cell_type"] == "code" and "source" in c:
            c["source"] = [line + "\n" for line in c["source"]] if isinstance(c["source"][0], str) and not c["source"][0].endswith("\n") else c["source"]
        if c["cell_type"] == "markdown" and "source" in c:
            src = c["source"]
            if src and isinstance(src[0], str) and not src[0].endswith("\n"):
                c["source"] = [line + "\n" for line in src]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_path = root / "soundbar_model_agent.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("생성 완료:", out_path)


if __name__ == "__main__":
    main()
