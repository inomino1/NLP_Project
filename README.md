## 사운드바 모델 매핑 Agent

`raw_data/HDMI_BT_Log.csv`의 관측 기기명(`NAME1~4`, 옵션 `NAME_BT`)을 이용해, `soundbar_list.py`의 **표준 canonical 모델명**(예: `LG S90TY`)으로 매핑합니다. 불확실한 경우 **UNKNOWN**을 반환하도록 설계되어 있습니다.

### 프로젝트 구조

- `raw_data/` - RAW 데이터 (HDMI_BT_Log.csv)
- `output/` - 출력 파일 (pred.csv, pred_baseline.csv, labels.csv, sweep.csv)

### 주요 구성 요소

- 사운드바 DB 로드/정규화
- **후보 문자열 생성**: 로그 행(NAME[1,2,3,4], BRAND[1,2,3,4], NAME_BT)에서 검색에 사용할 질의 문자열을 추출합니다. placeholder(HDMI, AV 등) 제거, 결측값(NAN) 필터링, 브랜드+모델명 조합 생성, 중복 제거 등을 수행합니다.
- **임베딩 기반 후보 검색**: 전체 사운드바 모델 리스트 DB에 대해 sentence-transformers 임베딩의 코사인 유사도로 직접 검색하여 **Top-5 후보**만 반환합니다. 
- 룰/패턴 기반 비사운드바 기기 탐지 (device_type, NER 미사용)
- UNKNOWN 결정 : 가장 높은 후보 점수(top1)가 임계치(threshold)보다 낮으면 UNKNOWN 으로 간주하는 작업
- 평가(Accuracy/F1/Top-k/MRR)

### 라벨 포맷(평가용)

`output/labels.csv` (UTF-8 권장)

- **true_model**: 정답 canonical 모델명 또는 `UNKNOWN`
- pred와 동일한 행 수·순서여야 합니다.

### 실행 예시

예측 (기본: raw_data/HDMI_BT_Log.csv → output/pred.csv):

```bash
python -m src.cli predict --input "raw_data/HDMI_BT_Log.csv" --db "soundbar_list.py" --output "output/pred.csv"
```

baseline 예측 (기본: output/pred_baseline.csv):

```bash
python -m src.cli baseline-predict --input "raw_data/HDMI_BT_Log.csv" --db "soundbar_list.py" --output "output/pred_baseline.csv"
```

평가 (기본: output/pred.csv, output/labels.csv):

```bash
python -m src.cli evaluate --pred "output/pred.csv" --labels "output/labels.csv"
```

threshold 탐색:

```bash
python -m src.cli sweep --pred "output/pred.csv" --labels "output/labels.csv" --out "output/sweep.csv"
```

