# 실험 기록

실험을 돌릴 때마다 한 줄씩 추가합니다. 지표는 test set, 임계값 0.5, 폭력=양성 기준.

## 정확도 지표

| 날짜 | 브랜치 | 모델 | 설정 (T, epochs 등) | Recall | Precision | F1 | Accuracy | 메모 |
|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | exp/01-baseline | MobileNetV2+LSTM | T=16, epochs=50(early stop) |  |  |  |  |  |
| YYYY-MM-DD | exp/01-baseline | MobileNetV2+TCN | T=16, epochs=50(early stop) |  |  |  |  |  |

## 속도 지표

| 날짜 | 측정 기기 | 모델 | Inference Time (ms, 중앙값) | p95 (ms) | FPS | 메모 |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | Colab T4 (참고용) | MobileNetV2+LSTM |  |  |  |  |
| YYYY-MM-DD | Colab T4 (참고용) | MobileNetV2+TCN |  |  |  |  |
| YYYY-MM-DD | (배포 기기) | MobileNetV2+LSTM |  |  |  |  |
| YYYY-MM-DD | (배포 기기) | MobileNetV2+TCN |  |  |  |  |

## 실험 노트
- (무엇을 바꿨고, 왜 그렇게 했고, 결과가 어땠는지 간단히)
