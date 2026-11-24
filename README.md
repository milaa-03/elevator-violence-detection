# 엘리베이터 폭력 감지 (Elevator Violence Detection)

엘리베이터 안에서 발생하는 폭력을 감지하고, 감지 시 가까운 층에 정차·문 열림·자동 신고를 하는 시스템의 **1차 과제: 모델 평가 지표 산출** 저장소입니다.

## 과제 목표
- Vision model: **MobileNetV2** (프레임별 특징 추출)
- Time-analysis model: **LSTM**, **TCN**
- 평가 지표: **Recall, Precision, F1-score, Accuracy, FPS, Inference Time**
- 데이터셋: **SCVD** (Smart-City CCTV Violence Detection) — `docs/datasets.md` 참고

## 저장소 구조
```
├─ rvd_pipeline.py              # extract / train / speed 3단계 스크립트
├─ notebooks/01_colab_train.ipynb  # Colab에서 순서대로 실행하는 노트북
├─ docs/
│  ├─ GUIDE.md                  # 처음부터 끝까지 단계별 가이드 (Colab + GitHub)
│  ├─ datasets.md               # 데이터 출처, 라이선스, 클래스 매핑
│  └─ experiments.md            # 실험 기록표 (실험할 때마다 한 줄 추가)
├─ results/                     # metrics.json, speed.json (실험 결과)
└─ requirements.txt
```

## 빠른 시작
1. `notebooks/01_colab_train.ipynb`를 Colab에서 열고 GPU 런타임으로 설정
2. 셀을 위에서부터 순서대로 실행
3. 결과(`results/*.json`)를 `docs/experiments.md`에 기록하고 커밋

자세한 설명은 [docs/GUIDE.md](docs/GUIDE.md).

## 지표 정의
| 지표 | 정의 |
|---|---|
| Recall / Precision / F1 / Accuracy | 폭력(1)을 양성(positive)으로 보고 test set에서 임계값 0.5로 계산 |
| Inference Time | 16프레임 클립 1개를 판정하는 시간(ms): 전처리 + MobileNetV2 + 시계열 모델 (중앙값, p95) |
| FPS | 16 ÷ 클립 추론 시간(초). 처리하는 프레임 수 기준 |

> ⚠️ Colab(GPU)에서 잰 속도는 참고용입니다. 최종 FPS/추론시간은 **실제 배포 기기**에서 `speed` 단계를 다시 실행해 기록하세요.
