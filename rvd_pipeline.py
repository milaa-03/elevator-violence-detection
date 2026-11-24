"""
엘리베이터 폭력 감지 - MobileNetV2 + (LSTM | TCN) 학습 및 평가 파이프라인

데이터 구조: --data 아래에 어떤 하위 폴더 구조든 상관없음 (재귀 탐색).
  경로에 NEG_KEYWORDS(normal, non-violence 등)가 들어가면 0(비폭력), 아니면 1(폭력).
  예) SCVD: Violence / Weaponized / Normal 폴더 -> Normal=0, 나머지=1

사용법:
  python rvd_pipeline.py extract --data data --out feats        # 1) 영상 -> MobileNetV2 특징
  python rvd_pipeline.py train   --feats feats --out results    # 2) LSTM/TCN 학습 + 지표
  python rvd_pipeline.py speed   --data data --out results      # 3) FPS / 추론시간 (end-to-end)
"""
import argparse, glob, json, os, time
import numpy as np
import cv2

T = 16          # 클립당 샘플링 프레임 수
SIZE = 224      # MobileNetV2 입력 크기
SEED = 42


# ---------------------------------------------------------------- 공통
def sample_frames(path, t=T, size=SIZE):
    """영상에서 균등 간격으로 t개 프레임을 뽑아 (t,size,size,3) RGB uint8 반환."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(n - 1, 0), t).astype(int).tolist())
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            f = cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB)
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"프레임을 읽지 못함: {path}")
    while len(frames) < t:              # 짧은 영상은 마지막 프레임 반복
        frames.append(frames[-1])
    return np.stack(frames[:t])


def build_backbone():
    import tensorflow as tf
    return tf.keras.applications.MobileNetV2(
        include_top=False, weights=(os.environ.get("RVD_WEIGHTS", "imagenet") or None), pooling="avg",
        input_shape=(SIZE, SIZE, 3))


def preprocess(frames):
    import tensorflow as tf
    return tf.keras.applications.mobilenet_v2.preprocess_input(frames.astype("float32"))


def build_lstm(t=T, d=1280):
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((t, d)), L.LSTM(128), L.Dropout(0.5),
        L.Dense(64, activation="relu"), L.Dense(1, activation="sigmoid")], name="lstm")


def build_tcn(t=T, d=1280, filters=64, kernel=3, dilations=(1, 2, 4)):
    """Dilated causal Conv1D 잔차 블록을 쌓은 간단한 TCN."""
    import tensorflow as tf
    L = tf.keras.layers
    x_in = L.Input((t, d))
    x = L.Conv1D(filters, 1)(x_in)
    for dl in dilations:
        h = L.Conv1D(filters, kernel, padding="causal", dilation_rate=dl, activation="relu")(x)
        h = L.SpatialDropout1D(0.2)(h)
        h = L.Conv1D(filters, kernel, padding="causal", dilation_rate=dl, activation="relu")(h)
        h = L.SpatialDropout1D(0.2)(h)
        x = L.Activation("relu")(L.Add()([x, h]))
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(64, activation="relu")(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(x_in, out, name="tcn")


# ---------------------------------------------------------------- 1) 특징 추출
NEG_KEYWORDS = ("nonviolen", "non-violen", "non_violen", "non violen", "normal", "nonfight", "non-fight")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def label_of(path, neg=NEG_KEYWORDS):
    """경로 어디에든 비폭력 키워드가 있으면 0, 아니면 1(폭력). 'nonviolence'가 'violence'를 포함하므로 비폭력부터 검사."""
    low = path.lower()
    return 0 if any(k in low for k in neg) else 1


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(a.data, "**", "*"), recursive=True)
                   if p.lower().endswith(VIDEO_EXT))
    if not paths:
        raise SystemExit(f"영상을 찾지 못함: {a.data}")
    neg = tuple(a.neg.split(",")) if a.neg else NEG_KEYWORDS
    labels = [label_of(os.path.relpath(p, a.data), neg) for p in paths]
    print(f"영상 {len(paths)}개 | 비폭력(0) {labels.count(0)}개 | 폭력(1) {labels.count(1)}개")
    if a.dry_run:                       # 라벨링이 맞는지 확인만
        for p, l in list(zip(paths, labels))[:: max(1, len(paths) // 15)]:
            print(l, os.path.relpath(p, a.data))
        return
    bb = build_backbone()
    X, y, names, skipped = [], [], [], []
    for i, (p, l) in enumerate(zip(paths, labels)):
        try:
            f = bb.predict(preprocess(sample_frames(p)), verbose=0)   # (T,1280)
        except Exception as e:           # 깨진 영상은 건너뜀
            skipped.append(p); print("skip", p, e); continue
        X.append(f); y.append(l); names.append(os.path.relpath(p, a.data))
        if i % 20 == 0:
            print(f"{i + 1}/{len(paths)}")
    np.savez_compressed(os.path.join(a.out, "feats.npz"),
                        X=np.array(X, dtype="float32"), y=np.array(y), names=np.array(names))
    print("saved", len(y), "clips, skipped", len(skipped))


# ---------------------------------------------------------------- 2) 학습 + 평가
def group_of(name):
    """누적 누수 방지용 그룹. 같은 상위 파일명 접두(예: cam1_xxx)는 같은 그룹으로 취급.
    필요하면 촬영 장면/출처 기준으로 직접 바꾸세요."""
    base = os.path.splitext(os.path.basename(name))[0]
    return base          # 기본: 파일 하나가 한 그룹. 같은 원본에서 잘린 클립이 있다면 여기서 원본 ID로 묶으세요.


def split(names, y):
    from sklearn.model_selection import GroupShuffleSplit
    g = np.array([group_of(n) for n in names])
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    trval, te = next(gss.split(names, y, g))
    gss2 = GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
    tr, va = next(gss2.split(trval, y[trval], g[trval]))
    return trval[tr], trval[va], te


def metrics(y, prob, thr=0.5):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    p = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(recall=recall_score(y, p, zero_division=0),
                precision=precision_score(y, p, zero_division=0),
                f1=f1_score(y, p, zero_division=0),
                accuracy=accuracy_score(y, p), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def cmd_train(a):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(a.out, exist_ok=True)
    d = np.load(os.path.join(a.feats, "feats.npz"))
    X, y, names = d["X"], d["y"], d["names"]
    tr, va, te = split(names, y)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}")
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm}: 비폭력 {(y[ix] == 0).sum()} / 폭력 {(y[ix] == 1).sum()}")
    # 클래스 불균형 보정
    w = {c: len(tr) / (2 * max((y[tr] == c).sum(), 1)) for c in (0, 1)}
    results = {}
    for name, build in [("MobileNetV2+LSTM", build_lstm), ("MobileNetV2+TCN", build_tcn)]:
        m = build(t=X.shape[1], d=X.shape[2])
        m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)]
        m.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=a.epochs,
              batch_size=16, class_weight=w, callbacks=cb, verbose=2)
        m.save(os.path.join(a.out, name.replace("+", "_") + ".keras"))
        prob = m.predict(X[te], verbose=0).ravel()
        results[name] = metrics(y[te], prob)
        results[name]["params"] = int(m.count_params())
        print(name, results[name])
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- 3) 속도
def cmd_speed(a):
    """end-to-end(프레임 전처리 + MobileNetV2 + 시계열 모델) 추론 시간과 FPS.
    - Inference time: 16프레임 클립 1개를 판정하는 데 걸리는 시간(ms), 단계별 분해
    - FPS: (클립당 프레임 수) / (클립 추론 시간)  → 처리하는 프레임 수 기준
    반드시 라즈베리파이에서도 같은 스크립트로 다시 측정하세요."""
    import tensorflow as tf
    os.makedirs(a.out, exist_ok=True)
    vids = sorted(p for p in glob.glob(os.path.join(a.data, "**", "*"), recursive=True) if p.lower().endswith(VIDEO_EXT))[: a.n]
    clips = [sample_frames(v) for v in vids]
    bb = build_backbone()
    out = {}
    for name, build in [("MobileNetV2+LSTM", build_lstm), ("MobileNetV2+TCN", build_tcn)]:
        head = build()
        for c in clips[:3]:                                    # warm-up
            head.predict(bb.predict(preprocess(c), verbose=0)[None], verbose=0)
        t_pre, t_bb, t_head = [], [], []
        for c in clips:
            s = time.perf_counter(); x = preprocess(c); t_pre.append(time.perf_counter() - s)
            s = time.perf_counter(); f = bb(x, training=False).numpy(); t_bb.append(time.perf_counter() - s)
            s = time.perf_counter(); head(f[None], training=False).numpy(); t_head.append(time.perf_counter() - s)
        tot = np.array(t_pre) + np.array(t_bb) + np.array(t_head)
        out[name] = dict(
            clips=len(clips), frames_per_clip=T,
            preprocess_ms=float(np.median(t_pre) * 1e3),
            backbone_ms=float(np.median(t_bb) * 1e3),
            temporal_ms=float(np.median(t_head) * 1e3),
            inference_ms_median=float(np.median(tot) * 1e3),
            inference_ms_p95=float(np.percentile(tot, 95) * 1e3),
            fps=float(T / np.median(tot)))
        print(name, out[name])
    out["_env"] = dict(tf=tf.__version__, cpu_count=os.cpu_count(),
                       gpu=[g.name for g in tf.config.list_physical_devices("GPU")])
    with open(os.path.join(a.out, "speed.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--data", required=True); e.add_argument("--out", default="feats"); e.add_argument("--neg", default=""); e.add_argument("--dry-run", action="store_true")
    t = sp.add_parser("train"); t.add_argument("--feats", default="feats"); t.add_argument("--out", default="results"); t.add_argument("--epochs", type=int, default=50)
    s = sp.add_parser("speed"); s.add_argument("--data", required=True); s.add_argument("--out", default="results"); s.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "speed": cmd_speed}[a.cmd](a)
