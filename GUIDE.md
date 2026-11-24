# 처음부터 끝까지 가이드 (Colab 학습 + GitHub 기록)

## 0. 먼저 이해할 것

| 항목 | 내용 |
|---|---|
| 하는 일 | 영상 → MobileNetV2로 프레임 특징 추출 → LSTM / TCN으로 폭력 여부 분류 → 지표 6개 산출 |
| 어디서 | 학습·정확도 지표는 **Colab(GPU)**, FPS·추론시간은 **실제 배포 기기** |
| 결과물 | `results/metrics.json`, `results/speed.json`, `docs/experiments.md` 표 |

---

## 1. GitHub repo 준비 (한 번만)

1. GitHub에서 새 repo 생성 (예: `elevator-violence-detection`). 팀원 모두 Collaborator로 초대.
2. 이 폴더의 파일을 repo에 올립니다.
   ```bash
   cd elevator-violence-detection
   git init
   git add .
   git commit -m "init: 프로젝트 뼈대, 파이프라인, 노트북 추가"
   git branch -M main
   git remote add origin https://github.com/<팀>/elevator-violence-detection.git
   git push -u origin main
   ```
3. `.gitignore`가 영상(`*.mp4`), 특징(`*.npz`), 모델(`*.keras`)을 막아 줍니다. **영상 데이터는 절대 올리지 마세요.**

## 2. Colab 준비

1. Google Drive에 `rvd` 폴더를 만들어 둡니다 (특징 파일 백업용).
2. [colab.research.google.com](https://colab.research.google.com) → **파일 → 노트 업로드**로 `notebooks/01_colab_train.ipynb` 열기
   (또는 GitHub 탭에서 repo의 노트북 열기).
3. **런타임 → 런타임 유형 변경 → T4 GPU** 선택.
4. 노트북 첫 셀의 `REPO_URL`을 팀 repo 주소로 바꿉니다.

## 3. Kaggle 데이터 받기 (SCVD)

1. Kaggle → 프로필 → **Settings → API → Create New Token** → `kaggle.json` 다운로드.
2. 노트북의 Kaggle 셀 실행 → `kaggle.json` 업로드 → 자동 다운로드·압축 해제.
3. **폴더 구조 확인 셀**의 출력을 보고 `docs/datasets.md`에 기록합니다.

## 4. 라벨링 확인 (중요)

`rvd_pipeline.py`는 경로에 `normal`, `non-violence` 같은 단어가 있으면 **비폭력(0)**, 없으면 **폭력(1)** 으로 봅니다. 즉 `Violence`, `Weaponized Violence`는 모두 폭력(1)입니다.

- 노트북의 **dry-run 셀**이 영상 개수와 샘플 라벨을 보여줍니다. 비폭력/폭력 개수가 데이터 설명과 맞는지 확인하세요.
- 폴더 이름이 다르면(예: `Safe`, `Fight`) `--neg "safe"`처럼 비폭력 키워드를 지정합니다.
- 무기 폭력을 따로 보고 싶다면 `docs/datasets.md`에 3클래스로 쪼갠 실험을 후속 과제로 적어 두세요.

## 5. 특징 추출 → 학습 → 지표

노트북 셀 순서대로 실행합니다.

| 단계 | 명령 | 산출물 |
|---|---|---|
| extract | `python rvd_pipeline.py extract --data /content/scvd --out feats` | `feats/feats.npz` |
| train | `python rvd_pipeline.py train --feats feats --out results` | `results/metrics.json`, `*.keras` |
| speed | `python rvd_pipeline.py speed --data /content/scvd --out results` | `results/speed.json` |

- extract가 가장 오래 걸립니다 (영상 수에 비례). 끝나면 `feats.npz`를 Drive에 백업하는 셀이 있으니 실행해 두세요. 런타임이 끊겨도 다시 추출하지 않아도 됩니다.
- train은 특징만 쓰므로 빠릅니다. 하이퍼파라미터(`--epochs` 등)를 바꿔 여러 번 돌려도 됩니다.

## 6. 속도(FPS, 추론시간) 측정

- Colab 결과는 **참고용**으로만 적습니다 (GPU라 실제 기기와 다름).
- 최종 수치는 실제 배포 기기(라즈베리파이 등)에서:
  ```bash
  pip install tensorflow opencv-python-headless scikit-learn
  python rvd_pipeline.py speed --data <샘플 영상 폴더> --out results
  ```
- 기기 종류와 CPU/GPU 정보를 `experiments.md` 메모에 꼭 적으세요. (`speed.json`의 `_env`에도 기록됩니다.)

## 7. 결과를 GitHub에 기록

### 7-1. 기록 규칙
- 실험 1회 = 브랜치 1개 = 커밋 몇 개 = `experiments.md` 한 줄
- 브랜치 이름: `exp/01-baseline`, `exp/02-more-epochs`, `exp/03-threshold`
- 커밋 메시지 접두어:

| 접두어 | 의미 | 예 |
|---|---|---|
| `data:` | 데이터 수집/정리 | `data: SCVD 다운로드 방법과 클래스 매핑 기록` |
| `train:` | 학습 코드/설정 | `train: LSTM/TCN baseline 학습` |
| `eval:` | 평가 결과 | `eval: baseline metrics.json 추가` |
| `docs:` | 문서 | `docs: experiments 표에 baseline 기록` |

### 7-2. 실험 하나를 기록하는 순서
1. `git checkout -b exp/01-baseline`
2. 노트북 실행 후 `results/metrics.json`, `results/speed.json` 확보
3. `docs/experiments.md` 표에 한 줄 추가 (날짜, 모델, 설정, 지표, 메모)
4. 커밋 & 푸시
   ```bash
   git add results/*.json docs/experiments.md notebooks/
   git commit -m "eval: baseline 결과 추가"
   git push -u origin exp/01-baseline
   ```
5. GitHub에서 Pull Request 생성 → 팀원 확인 → `main`에 merge

### 7-3. Colab에서 바로 push하는 방법
노트북 마지막 셀이 담당합니다.
1. GitHub → Settings → Developer settings → **Personal access tokens (fine-grained)** → 해당 repo에 `Contents: Read and write` 권한으로 생성
2. Colab 왼쪽 **🔑 보안 비밀(Secrets)** → 이름 `GITHUB_TOKEN`, 값에 토큰 붙여넣기, 노트북 액세스 켜기
3. 마지막 셀의 `BRANCH`, `MSG`를 바꿔 실행

> 토큰은 노트북 코드에 직접 붙여넣지 마세요. 노트북을 repo에 올리면 토큰이 공개됩니다.

(대안) 결과 json만 내려받아 로컬에서 커밋해도 됩니다.

## 8. 자주 막히는 문제

| 증상 | 해결 |
|---|---|
| `영상을 찾지 못함` | `--data` 경로 확인. 압축이 한 단계 더 풀려 있는지 `find`로 확인 |
| 비폭력/폭력 개수가 이상함 | dry-run으로 확인 후 `--neg` 키워드 조정 |
| 점수가 지나치게 높음(예: 99%+) | 같은 원본에서 잘린 클립이 train/test에 섞였을 가능성. `group_of()`를 원본 ID 기준으로 수정 |
| Colab 런타임이 끊김 | `feats.npz`를 Drive에서 불러와 train부터 다시 실행 |
| 메모리 부족 | extract는 영상을 하나씩 처리하므로 보통 문제없음. 런타임 재시작 후 재실행 |
| test에 한 클래스만 있음 | 데이터가 적을 때 발생. `SEED`를 바꾸거나 분할을 직접 지정 |

## 9. 보고서에 넣을 한계점 (메모)
- SCVD는 일반 CCTV 영상이며 **엘리베이터 내부 영상이 아님** → 도메인 차이를 한계로 명시
- 지표는 임계값 0.5 기준. 서비스에서는 **오탐(자동 정차·신고)** 비용과 **미탐** 비용을 함께 고려해 임계값을 정해야 함
- 속도는 측정 기기에 따라 크게 달라지므로 기기 사양과 함께 기록
