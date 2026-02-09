"""
커맨드라인 엔트리포인트입니다.

예시:
- 예측:
  python -m src.cli predict --input "raw_data/HDMI_BT_Log.csv" --db "soundbar_list.py" --output "output/pred.csv"

- baseline 예측:
  python -m src.cli baseline-predict --input "raw_data/HDMI_BT_Log.csv" --db "soundbar_list.py" --output "output/pred_baseline.csv"

- 평가:
  python -m src.cli evaluate --pred "output/pred.csv" --labels "output/labels.csv"

- threshold 탐색:
  python -m src.cli sweep --pred "output/pred.csv" --labels "output/labels.csv" --out "output/sweep.csv"
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .agent import AgentConfig, SoundbarModelAgent
from .baseline import BaselineConfig, BaselineSoundbarMatcher
from .data_loader import load_hdmi_bt_log_csv, load_labels_csv
from .evaluate import evaluate_predictions, sweep_thresholds


def _path(x: str) -> Path:
    """argparse에서 Path 변환을 수행합니다."""
    return Path(x)


def _load_training_override(path: Path, encoding: str = "utf-8") -> dict[int, str]:
    """
    training CSV에서 row_id -> true_model 매핑을 로드합니다.
    true_model이 UNKNOWN/비어 있으면 제외합니다.
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
    except Exception as e:  # noqa: BLE001
        raise OSError(f"Training CSV 로드 실패: {path}") from e
    if "row_id" not in df.columns or "true_model" not in df.columns:
        raise ValueError("Training CSV는 row_id, true_model 컬럼을 포함해야 합니다.")
    lookup: dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            rid = int(r["row_id"])
        except (ValueError, TypeError):
            continue
        tm = str(r.get("true_model", "")).strip()
        if tm and tm.upper() != "UNKNOWN":
            lookup[rid] = tm
    return lookup


def _predict_chunk(args_tuple: tuple) -> Any:
    """워커용: 청크 예측. (chunk_df, db_path, config_kwargs, per_source, row_id_col)"""
    chunk_df, db_path, config_kwargs, per_source, row_id_col = args_tuple
    from .agent import AgentConfig, SoundbarModelAgent
    config = AgentConfig(**config_kwargs)
    agent = SoundbarModelAgent(db_path, config=config)
    if per_source:
        return agent.batch_predict_per_source(chunk_df, row_id_col=row_id_col)
    return agent.batch_predict(chunk_df, row_id_col=row_id_col)


def cmd_predict(args: argparse.Namespace) -> int:
    """predict 서브커맨드를 실행합니다."""
    t_start = time.perf_counter()
    try:
        log_res = load_hdmi_bt_log_csv(args.input, encoding=args.encoding)
        # accuracy_mode 기본값: True
        # - 별도 옵션을 주지 않으면 항상 정확도 우선 모드로 동작
        # - 호환성을 위해 --accuracy, --no-accuracy 둘 다 지원하며,
        #   --no-accuracy가 지정되면 accuracy_mode=False로 강제합니다.
        accuracy_mode: bool = True
        if getattr(args, "no_accuracy", False):
            accuracy_mode = False
        elif getattr(args, "accuracy", False):
            accuracy_mode = True

        config_kwargs = {
            "top_n_lexical": args.top_n_lexical,
            "top_k": args.top_k,
            "embedding_model_id": args.embedding_model_id,
            "use_embeddings": not args.no_embeddings,
            "include_bt": not args.no_bt,
            "accept_score_threshold": args.accept_threshold,
            "margin_threshold": args.margin_threshold,
            "accuracy_mode": accuracy_mode,
            "min_cosine_similarity": None if getattr(args, "no_cosine_gate", False) else getattr(args, "min_cosine_similarity", 0.85),
        }
        workers = getattr(args, "workers", 1) or 1
        per_source = getattr(args, "per_source", False)

        if workers <= 1:
            agent = SoundbarModelAgent(args.db, config=AgentConfig(**config_kwargs))
            pred_df = (
                agent.batch_predict_per_source(log_res.df)
                if per_source
                else agent.batch_predict(log_res.df)
            )
        else:
            # chunk에 원본 row_id를 부여하여 병렬 처리 후에도 올바른 row_id 유지
            df_for_chunk = log_res.df.copy()
            if "row_id" not in df_for_chunk.columns:
                df_for_chunk["row_id"] = range(len(df_for_chunk))
            row_id_col = "row_id"
            n = len(df_for_chunk)
            chunk_size = max(1, (n + workers - 1) // workers)
            chunks = [
                df_for_chunk.iloc[i : i + chunk_size]
                for i in range(0, n, chunk_size)
            ]
            args_tuples = [
                (chunk, args.db, config_kwargs, per_source, row_id_col)
                for chunk in chunks
            ]
            # chunk 완료 순서가 아닌 원본 순서로 결과 병합
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_predict_chunk, t) for t in args_tuples]
                dfs = [f.result() for f in futures]
            pred_df = pd.concat(dfs, ignore_index=True)
        training_path = getattr(args, "training", None)
        if training_path is not None:
            override = _load_training_override(training_path, encoding=args.encoding)
            row_col = "row_id" if "row_id" in pred_df.columns else "id"
            if row_col not in pred_df.columns:
                raise ValueError("예측 결과에 row_id/id 컬럼이 없습니다.")
            row_vals = pd.to_numeric(pred_df[row_col], errors="coerce")
            for rid, true_model in override.items():
                mask = row_vals == rid
                pred_df.loc[mask, "predicted_model"] = true_model
        if args.output:
            try:
                pred_df.to_csv(args.output, index=False, encoding="utf-8")
            except OSError as e:
                raise OSError(f"예측 CSV 저장 실패: {args.output}") from e
        else:
            print(pred_df.head(20).to_string(index=False))
        elapsed = time.perf_counter() - t_start
        print(f"소요 시간: {elapsed:.1f}초")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] predict 실패: {e}")
        return 2


def cmd_baseline_predict(args: argparse.Namespace) -> int:
    """baseline-predict 서브커맨드를 실행합니다."""
    try:
        log_res = load_hdmi_bt_log_csv(args.input, encoding=args.encoding)
        matcher = BaselineSoundbarMatcher(
            args.db,
            config=BaselineConfig(
                include_bt=not args.no_bt,
                accept_score_threshold=float(args.accept_threshold),
                top_k=int(args.top_k),
                use_brand_filter=not args.no_brand_filter,
                prefilter_top_n=int(args.prefilter_top_n),
            ),
        )
        pred_df = (
            matcher.batch_predict_per_source(log_res.df)
            if getattr(args, "per_source", False)
            else matcher.batch_predict(log_res.df)
        )
        training_path = getattr(args, "training", None)
        if training_path is not None:
            override = _load_training_override(training_path, encoding=args.encoding)
            row_col = "row_id" if "row_id" in pred_df.columns else "id"
            if row_col not in pred_df.columns:
                raise ValueError("예측 결과에 row_id/id 컬럼이 없습니다.")
            row_vals = pd.to_numeric(pred_df[row_col], errors="coerce")
            for rid, true_model in override.items():
                mask = row_vals == rid
                pred_df.loc[mask, "predicted_model"] = true_model
        if args.output:
            try:
                pred_df.to_csv(args.output, index=False, encoding="utf-8")
            except OSError as e:
                raise OSError(f"예측 CSV 저장 실패: {args.output}") from e
        else:
            print(pred_df.head(20).to_string(index=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] baseline-predict 실패: {e}")
        return 2


def _read_csv_with_fallback(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """인코딩 폴백으로 CSV를 로드합니다. utf-8 실패 시 utf-8-sig, cp949 등을 시도합니다."""
    encodings = [encoding, "utf-8-sig", "cp949", "latin-1"]
    seen: set[str] = set()
    last_err: Optional[Exception] = None
    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return pd.read_csv(path, encoding=enc)
        except FileNotFoundError:
            raise
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise OSError(f"예측 CSV 인코딩 오류: {path} (시도: {encodings})") from last_err


def cmd_evaluate(args: argparse.Namespace) -> int:
    """evaluate 서브커맨드를 실행합니다."""
    try:
        pred_df = _read_csv_with_fallback(args.pred, encoding=args.encoding)

        if args.labels is not None:
            labels_res = load_labels_csv(args.labels, encoding=args.encoding)
            labels_df = labels_res.df
        elif "true_model" in pred_df.columns:
            labels_df = pred_df[["true_model"]].copy()
            if args.exclude_unlabeled:
                t = labels_df["true_model"].astype(str).str.strip().str.upper()
                labels_df = labels_df[(t != "") & (t != "NAN")].copy()
            if labels_df.empty:
                raise ValueError("라벨이 비어 있습니다. true_model 컬럼에 정답을 기재하거나 --exclude-unlabeled를 해제하세요.")
        else:
            raise ValueError("--labels를 지정하거나, 예측 CSV에 true_model 컬럼이 있어야 합니다.")

        metrics = evaluate_predictions(
            pred_df,
            labels_df,
            include_unknown_in_f1=not args.exclude_unknown,
            exclude_both_unknown=args.exclude_unknown,
        )
        print("## Evaluation report")
        print(f"- n: {metrics.n}")
        print(f"- accuracy: {metrics.accuracy:.4f}")
        print(f"- f1_micro: {metrics.f1_micro:.4f}")
        print(f"- f1_macro: {metrics.f1_macro:.4f}")
        print(f"- top1_recall: {metrics.top1_recall:.4f}")
        print(f"- top3_recall: {metrics.top3_recall:.4f}")
        print(f"- top5_recall: {metrics.top5_recall:.4f}")
        print(f"- mrr: {metrics.mrr:.4f}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] evaluate 실패: {e}")
        return 2


def cmd_sweep(args: argparse.Namespace) -> int:
    """sweep 서브커맨드를 실행합니다."""
    try:
        pred_df = _read_csv_with_fallback(args.pred, encoding=args.encoding)
        labels_res = load_labels_csv(args.labels, encoding=args.encoding)
        sweep_df = sweep_thresholds(pred_df, labels_res.df)

        if args.out:
            try:
                sweep_df.to_csv(args.out, index=False, encoding="utf-8")
            except OSError as e:
                raise OSError(f"sweep 결과 저장 실패: {args.out}") from e
            print(f"sweep 결과 저장: {args.out}")
        else:
            print(sweep_df.head(30).to_string(index=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] sweep 실패: {e}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    """
    CLI 파서를 생성합니다.

    Returns:
        ArgumentParser
    """
    p = argparse.ArgumentParser(prog="soundbar-model-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="HDMI_BT_Log.csv에서 사운드바 모델 예측")
    p_pred.add_argument("--input", type=_path, default=Path("raw_data/HDMI_BT_Log.csv"), help="입력 로그 CSV 경로 (기본: raw_data/HDMI_BT_Log.csv)")
    p_pred.add_argument("--db", type=_path, required=True, help="soundbar_list.py 경로")
    p_pred.add_argument("--output", type=_path, default=Path("output/pred.csv"), help="예측 결과 CSV 경로 (기본: output/pred.csv)")
    p_pred.add_argument("--encoding", type=str, default="utf-8", help="CSV 인코딩")
    p_pred.add_argument("--top-n-lexical", type=int, default=200, help="TF-IDF TopN")
    p_pred.add_argument("--top-k", type=int, default=20, help="최종 TopK")
    p_pred.add_argument("--embedding-model-id", type=str, default="BAAI/bge-small-en-v1.5", help="임베딩 모델 ID")
    p_pred.add_argument("--no-embeddings", action="store_true", help="임베딩 사용 비활성화")
    p_pred.add_argument("--no-bt", action="store_true", help="NAME_BT 후보 사용 비활성화")
    p_pred.add_argument("--accept-threshold", type=float, default=0.52, help="accept score threshold")
    p_pred.add_argument("--margin-threshold", type=float, default=0.05, help="top1-top2 margin threshold")
    p_pred.add_argument("--per-source", action="store_true", help="소스별(BT/HDMI) 독립 예측, 행당 최대 5개 출력")
    p_pred.add_argument("--accuracy", action="store_true", help="NER+임베딩 전용 모드(정확도 우선, TF-IDF 비사용). 기본값: 사용")
    p_pred.add_argument("--no-accuracy", action="store_true", help="정확도 모드 비활성화 (기존 TF-IDF+임베딩 하이브리드 사용)")
    p_pred.add_argument("--min-cosine-similarity", type=float, default=0.85, help="primary_query vs top1 직접 코사인 유사도 임계치. 미만이면 UNKNOWN")
    p_pred.add_argument("--no-cosine-gate", action="store_true", help="코사인 유사도 최종 검증 비활성화")
    p_pred.add_argument("--training", type=_path, default=None, help="training CSV(row_id, true_model) 경로. 해당 row_id의 예측을 정답으로 덮어씀")
    p_pred.add_argument("--workers", type=int, default=1, help="병렬 처리 워커 수 (1=직렬, 2 이상=멀티프로세스)")
    p_pred.set_defaults(func=cmd_predict)

    p_b = sub.add_parser("baseline-predict", help="룰/Levenshtein baseline으로 예측")
    p_b.add_argument("--input", type=_path, default=Path("raw_data/HDMI_BT_Log.csv"), help="입력 로그 CSV 경로 (기본: raw_data/HDMI_BT_Log.csv)")
    p_b.add_argument("--db", type=_path, required=True, help="soundbar_list.py 경로")
    p_b.add_argument("--output", type=_path, default=Path("output/pred_baseline.csv"), help="예측 결과 CSV 경로 (기본: output/pred_baseline.csv)")
    p_b.add_argument("--encoding", type=str, default="utf-8", help="CSV 인코딩")
    p_b.add_argument("--top-k", type=int, default=5, help="top-k 후보 개수")
    p_b.add_argument("--accept-threshold", type=float, default=0.68, help="accept score threshold")
    p_b.add_argument("--prefilter-top-n", type=int, default=250, help="Levenshtein 전 프리필터 후보 수")
    p_b.add_argument("--no-bt", action="store_true", help="NAME_BT 후보 사용 비활성화")
    p_b.add_argument("--no-brand-filter", action="store_true", help="브랜드 기반 후보 풀 축소 비활성화")
    p_b.add_argument("--per-source", action="store_true", help="소스별(BT/HDMI) 독립 예측, 행당 최대 5개 출력")
    p_b.add_argument("--training", type=_path, default=None, help="training CSV(row_id, true_model) 경로. 해당 row_id의 예측을 정답으로 덮어씀")
    p_b.set_defaults(func=cmd_baseline_predict)

    p_eval = sub.add_parser("evaluate", help="예측 CSV와 라벨 CSV로 평가")
    p_eval.add_argument("--pred", type=_path, default=Path("output/pred.csv"), help="예측 CSV 경로 (기본: output/pred.csv)")
    p_eval.add_argument("--labels", type=_path, default=None, help="라벨 CSV 경로 (생략 시 pred의 true_model 사용, 지정 시 output/labels.csv 등)")
    p_eval.add_argument("--encoding", type=str, default="utf-8", help="CSV 인코딩")
    p_eval.add_argument("--exclude-unknown", action="store_true", help="F1에서 UNKNOWN 제외")
    p_eval.add_argument("--exclude-unlabeled", action="store_true", help="true_model이 비어 있는 행 제외(정답 미기재 행 제외)")
    p_eval.set_defaults(func=cmd_evaluate)

    p_sweep = sub.add_parser("sweep", help="threshold 탐색(accept/margin)")
    p_sweep.add_argument("--pred", type=_path, default=Path("output/pred.csv"), help="예측 CSV 경로 (기본: output/pred.csv)")
    p_sweep.add_argument("--labels", type=_path, default=Path("output/labels.csv"), help="라벨 CSV 경로 (기본: output/labels.csv)")
    p_sweep.add_argument("--encoding", type=str, default="utf-8", help="CSV 인코딩")
    p_sweep.add_argument("--out", type=_path, default=Path("output/sweep.csv"), help="sweep 결과 저장 경로 (기본: output/sweep.csv)")
    p_sweep.set_defaults(func=cmd_sweep)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI 메인 함수입니다.

    Args:
        argv: 인자 리스트(테스트용). None이면 sys.argv 사용.

    Returns:
        종료 코드
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

