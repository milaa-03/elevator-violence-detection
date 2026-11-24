# 데이터셋

## SCVD (Smart-City CCTV Violence Detection Dataset)

| 항목 | 내용 |
|---|---|
| 출처 | https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd |
| 논문 | SSIVD-Net (arXiv 2207.12850) |
| 내용 | 유튜브에서 모은 실제 CCTV 영상 — 비폭력 / 폭력 / 무기 폭력 |
| 영상 수 | (다운로드 후 기재) |
| 폴더 구조 | (다운로드 후 기재) |
| 라이선스 | (Kaggle 페이지 License 항목 확인 후 기재) |
| 다운로드 일자 | (기재) |

## 이진 분류 매핑

| 원본 클래스 | 라벨 |
|---|---|
| Normal / Non-violence | 0 (비폭력) |
| Violence | 1 (폭력) |
| Weaponized Violence | 1 (폭력) |

> 실제 폴더 이름은 다운로드 후 확인해서 위 표를 수정하세요.

## 분할
- 방식: (예: 영상 단위 랜덤 분할 / 원본 ID 기준 그룹 분할 / 공식 Train·Test 폴더 사용)
- 비율: train / val / test = (기재)
- 시드: 42

## 주의
- 영상 파일은 repo에 올리지 않습니다 (`.gitignore`로 제외).
- 재배포 조건은 라이선스를 확인하세요.
