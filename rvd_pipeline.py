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
        X.append(f); y.append(l); names.append(os.path.relpath(p, a.data).replace("\\", "/"))
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


def split(names, y, ignore_test=False, block=0):
    """분할 방식
    - 기본: 경로에 Train / Test 폴더가 있으면 공식 분할 사용.
    - ignore_test=True: Test 폴더는 버리고(SCVD의 Test는 Train의 복사본이라 평가에 쓸 수 없음)
      Train 안에서만 그룹 단위로 나눔. 그룹 = (클래스 폴더, 파일 번호 // block).
      같은 원본에서 잘린 연속 번호 클립이 train/test로 갈라지는 것을 막기 위함. block=0이면 파일 하나가 한 그룹.
    - Train/Test 폴더가 없으면 파일 단위 랜덤 그룹 분할."""
    import re
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    parts = [set(p.lower() for p in str(n).replace("\\", "/").split("/")) for n in names]
    is_te = np.array(["test" in p for p in parts])
    is_tr = np.array(["train" in p for p in parts])
    if ignore_test:
        pool = np.where(is_tr)[0] if is_tr.any() else np.arange(len(names))
        def grp(n):
            n = str(n).replace("\\", "/")
            folder = n.rsplit("/", 1)[0]
            m = re.search(r"(\d+)\.[A-Za-z0-9]+$", n)
            k = int(m.group(1)) if m else hash(n)
            return f"{folder}#{k // block if block else k}"
        g = np.array([grp(names[i]) for i in pool])
        gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
        a, te = next(gss.split(pool, y[pool], g))
        gss2 = GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
        b, va = next(gss2.split(pool[a], y[pool[a]], g[a]))
        print(f"Train 폴더만 사용, 그룹 분할 (block={block}), 그룹 수 {len(set(g))}")
        return pool[a][b], pool[a][va], pool[te]
    if is_te.any() and is_tr.any():
        te = np.where(is_te)[0]
        trval = np.where(is_tr)[0]
        tr, va = train_test_split(trval, test_size=0.15, random_state=SEED, stratify=y[trval])
        print("공식 Train/Test 분할 사용")
        return tr, va, te
    g = np.array([group_of(n) for n in names])
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    trval, te = next(gss.split(names, y, g))
    gss2 = GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
    tr, va = next(gss2.split(trval, y[trval], g[trval]))
    print("Train/Test 폴더 없음 -> 랜덤 그룹 분할")
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
    tr, va, te = split(names, y, a.ignore_test, a.block)
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


# ---------------------------------------------------------------- 2-b) 외부 데이터 평가
def cmd_eval(a):
    """학습에 쓰지 않은 외부 데이터셋(feats.npz)으로 저장된 모델들을 평가.
    사전에: python rvd_pipeline.py extract --data <외부데이터 폴더> --out feats_ext"""
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    d = np.load(a.feats); X, y = d["X"], d["y"]
    print(f"외부 데이터 {len(y)}개 | 비폭력(0) {(y == 0).sum()} / 폭력(1) {(y == 1).sum()}")
    out = {}
    for mp in sorted(glob.glob(os.path.join(a.models, "*.keras"))):
        name = os.path.basename(mp)[:-6]
        m = tf.keras.models.load_model(mp)
        prob = m.predict(X, verbose=0).ravel()
        r = metrics(y, prob, a.thr)
        r["roc_auc"] = float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None
        r["mean_prob_nonviolence"] = float(prob[y == 0].mean()) if (y == 0).any() else None
        r["mean_prob_violence"] = float(prob[y == 1].mean()) if (y == 1).any() else None
        out[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"external_{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- 2-c) End-to-end 파인튜닝
def _tcn_on(x, filters=64, kernel=3, dilations=(1, 2, 4)):
    import tensorflow as tf
    L = tf.keras.layers
    x = L.Conv1D(filters, 1)(x)
    for dl in dilations:
        h = L.Conv1D(filters, kernel, padding="causal", dilation_rate=dl, activation="relu")(x)
        h = L.SpatialDropout1D(0.2)(h)
        h = L.Conv1D(filters, kernel, padding="causal", dilation_rate=dl, activation="relu")(h)
        h = L.SpatialDropout1D(0.2)(h)
        x = L.Activation("relu")(L.Add()([x, h]))
    return L.Dense(64, activation="relu")(L.GlobalAveragePooling1D()(x))


def build_e2e(kind, size=SIZE, t=T):
    """입력: uint8 프레임 (t,size,size,3) -> MobileNetV2(프레임별) -> LSTM|TCN -> 폭력 확률. 전처리 포함."""
    import tensorflow as tf
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights=(os.environ.get("RVD_WEIGHTS", "imagenet") or None),
        pooling="avg", input_shape=(size, size, 3))
    inp = L.Input((t, size, size, 3), dtype="uint8")
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)          # mobilenet_v2 preprocess_input 과 동일
    x = L.TimeDistributed(base)(x)
    if kind == "LSTM":
        x = L.LSTM(128)(x); x = L.Dropout(0.5)(x); x = L.Dense(64, activation="relu")(x)
    else:
        x = _tcn_on(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name=f"mobilenetv2_{kind.lower()}_e2e"), base


def _set_trainable(base, n):
    """MobileNetV2 학습 대상 설정. n=0: 전부 고정 / n>0: 마지막 n개 층만 / n<0: 전체 층(완전 학습).
    BatchNorm은 항상 고정(작은 데이터에서 통계가 흔들리는 것을 방지)."""
    import tensorflow as tf
    base.trainable = n != 0
    if n != 0:
        target = base.layers if n < 0 else base.layers[-n:]
        for l in base.layers:
            l.trainable = False
        for l in target:
            if not isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = True


def _augment(x, size):
    """클립 전체에 같은 변환(좌우반전, 밝기/대비, 랜덤 크롭 후 리사이즈)을 적용."""
    import tensorflow as tf
    x = tf.cast(x, tf.float32)
    x = tf.cond(tf.random.uniform(()) < 0.5, lambda: x[:, :, ::-1, :], lambda: x)
    x = tf.image.random_brightness(x, 25.0)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    ch = tf.random.uniform((), int(size * 0.8), size + 1, dtype=tf.int32)
    x = tf.image.random_crop(x, tf.stack([T, ch, ch, 3]))
    x = tf.image.resize(x, (size, size))
    return tf.cast(tf.clip_by_value(x, 0.0, 255.0), tf.uint8)


def _make_ds(F, y, idx, batch, train, size):
    import tensorflow as tf
    def gen():
        order = np.random.permutation(idx) if train else idx
        for i in order:
            yield F[i], np.float32(y[i])
    sig = (tf.TensorSpec((T, size, size, 3), tf.uint8), tf.TensorSpec((), tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if train:
        ds = ds.map(lambda x, l: (_augment(x, size), l), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def _breakdown(names, y, prob, keys, thr=0.5):
    """Test 를 출처별로 나눠 지표 계산. keys: 경로에 포함될 키워드 목록(소문자). 어느 키에도 안 걸리면 'other'."""
    from sklearn.metrics import roc_auc_score
    names = np.array([str(n).lower() for n in names])
    out = {}
    assigned = np.zeros(len(names), dtype=bool)
    groups = []
    for k in keys:
        m = np.array([k in n for n in names]) & ~assigned
        groups.append((k, m)); assigned |= m
    groups.append(("other", ~assigned))
    for k, m in groups:
        if m.sum() == 0:
            continue
        r = metrics(y[m], prob[m], thr)
        r["n"] = int(m.sum())
        r["roc_auc"] = float(roc_auc_score(y[m], prob[m])) if len(set(y[m])) == 2 else None
        out[k] = r
    return out


def _save_preds(path, names, y, prob):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "label", "prob", "pred_at_0.5"])
        for n, l, p in zip(names, y, prob):
            w.writerow([n, int(l), round(float(p), 4), int(p >= 0.5)])


def cmd_finetune(a):
    """MobileNetV2 + LSTM/TCN 을 한 모델로 학습. 1단계: MobileNetV2 고정하고 헤드만, 2단계: 마지막 N개 층 해제 + 낮은 학습률."""
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, f"frames_{a.size}.npz")
    if os.path.exists(cache):
        d = np.load(cache); F, y, names = d["F"], d["y"], d["names"]
        print("프레임 캐시 사용:", cache)
    else:
        paths = sorted(p for p in glob.glob(os.path.join(a.data, "**", "*"), recursive=True) if p.lower().endswith(VIDEO_EXT))
        neg = tuple(a.neg.split(",")) if a.neg else NEG_KEYWORDS
        Fl, yl, names = [], [], []
        for i, p in enumerate(paths):
            try:
                Fl.append(sample_frames(p, T, a.size))
            except Exception as e:
                print("skip", p, e); continue
            rel = os.path.relpath(p, a.data).replace("\\", "/")
            yl.append(label_of(rel, neg)); names.append(rel)
            if i % 50 == 0:
                print(f"프레임 추출 {i + 1}/{len(paths)}")
        F, y, names = np.stack(Fl), np.array(yl), np.array(names)
        np.savez(cache, F=F, y=y, names=names)
        print("프레임 캐시 저장:", cache, F.shape)
    tr, va, te = split(names, y, a.ignore_test, a.block)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}")
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm}: 비폭력 {(y[ix] == 0).sum()} / 폭력 {(y[ix] == 1).sum()}")
    w = {c: len(tr) / (2 * max((y[tr] == c).sum(), 1)) for c in (0, 1)}
    ds_tr = _make_ds(F, y, tr, a.batch, True, a.size)
    ds_va = _make_ds(F, y, va, a.batch, False, a.size)
    ds_te = _make_ds(F, y, te, a.batch, False, a.size)
    results = {}
    for kind in ["LSTM", "TCN"]:
        m, base = build_e2e(kind, a.size)
        _set_trainable(base, 0)
        m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])
        print(f"[{kind}] 1단계: MobileNetV2 고정, 헤드만 {a.warmup} epoch")
        m.fit(ds_tr, validation_data=ds_va, epochs=a.warmup, class_weight=w, verbose=2)
        _set_trainable(base, a.unfreeze)
        m.compile(tf.keras.optimizers.Adam(a.lr), "binary_crossentropy", metrics=["accuracy"])
        print(f"[{kind}] 2단계: MobileNetV2 " + ("전체 층" if a.unfreeze < 0 else f"마지막 {a.unfreeze}개 층") + f" 해제, lr={a.lr}")
        cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)]
        m.fit(ds_tr, validation_data=ds_va, epochs=a.epochs, class_weight=w, callbacks=cb, verbose=2)
        m.save(os.path.join(a.out, f"E2E_MobileNetV2_{kind}.keras"))
        prob = m.predict(ds_te, verbose=0).ravel()
        r = metrics(y[te], prob)
        r["roc_auc"] = float(roc_auc_score(y[te], prob)) if len(set(y[te])) == 2 else None
        r["params"] = int(m.count_params())
        keys = [k.strip().lower() for k in a.sources.split(",") if k.strip()]
        if keys:
            r["by_source"] = _breakdown(names[te], y[te], prob, keys)
        _save_preds(os.path.join(a.out, f"preds_{kind}.csv"), names[te], y[te], prob)
        results[f"E2E MobileNetV2+{kind}"] = r
        print(f"E2E MobileNetV2+{kind}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "by_source"})
        for k, v in r.get("by_source", {}).items():
            print(f"   [{k}] n={v['n']}", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "n"})
    with open(os.path.join(a.out, "metrics_finetune.json"), "w", encoding="utf-8") as f:
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


def cmd_speed_e2e(a):
    """end-to-end 모델(영상 프레임 입력)의 추론 시간/FPS. 전처리(프레임 읽기+리사이즈)와 모델 추론을 나눠서 측정."""
    import tensorflow as tf
    os.makedirs(a.out, exist_ok=True)
    m = tf.keras.models.load_model(a.model)
    size = int(m.input_shape[2])
    vids = sorted(p for p in glob.glob(os.path.join(a.data, "**", "*"), recursive=True) if p.lower().endswith(VIDEO_EXT))[: a.n]
    t_read, t_model = [], []
    for v in vids[:3]:
        m(sample_frames(v, T, size)[None], training=False)          # warm-up
    for v in vids:
        s0 = time.perf_counter(); x = sample_frames(v, T, size)[None]; t_read.append(time.perf_counter() - s0)
        s0 = time.perf_counter(); m(x, training=False).numpy(); t_model.append(time.perf_counter() - s0)
    tot = np.array(t_read) + np.array(t_model)
    r = dict(model=os.path.basename(a.model), clips=len(vids), frames_per_clip=T,
             read_resize_ms=float(np.median(t_read) * 1e3), model_ms=float(np.median(t_model) * 1e3),
             inference_ms_median=float(np.median(tot) * 1e3), inference_ms_p95=float(np.percentile(tot, 95) * 1e3),
             fps=float(T / np.median(tot)), fps_model_only=float(T / np.median(t_model)),
             env=dict(tf=tf.__version__, cpu_count=os.cpu_count(), gpu=[g.name for g in tf.config.list_physical_devices("GPU")]))
    print(r)
    with open(os.path.join(a.out, "speed_" + os.path.basename(a.model).replace(".keras", "") + ".json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--data", required=True); e.add_argument("--out", default="feats"); e.add_argument("--neg", default=""); e.add_argument("--dry-run", action="store_true")
    t = sp.add_parser("train"); t.add_argument("--feats", default="feats"); t.add_argument("--out", default="results"); t.add_argument("--epochs", type=int, default=50); t.add_argument("--ignore-test", action="store_true"); t.add_argument("--block", type=int, default=0)
    v = sp.add_parser("eval"); v.add_argument("--feats", required=True); v.add_argument("--models", default="results"); v.add_argument("--out", default="results"); v.add_argument("--tag", default="external"); v.add_argument("--thr", type=float, default=0.5)
    ft = sp.add_parser("finetune"); ft.add_argument("--data", required=True); ft.add_argument("--out", default="results_ft"); ft.add_argument("--size", type=int, default=224)
    ft.add_argument("--epochs", type=int, default=30); ft.add_argument("--warmup", type=int, default=3); ft.add_argument("--unfreeze", type=int, default=-1, help="-1: MobileNetV2 전체 학습, N: 마지막 N개 층, 0: 고정")
    ft.add_argument("--lr", type=float, default=1e-5); ft.add_argument("--batch", type=int, default=4); ft.add_argument("--neg", default="")
    ft.add_argument("--ignore-test", action="store_true"); ft.add_argument("--block", type=int, default=0)
    ft.add_argument("--sources", default="rlvd", help="Test 를 출처별로 나눌 경로 키워드(쉼표 구분, 소문자). 예: rlvd. 빈 문자열이면 생략")
    se = sp.add_parser("speed-e2e"); se.add_argument("--model", required=True); se.add_argument("--data", required=True); se.add_argument("--out", default="results_ft"); se.add_argument("--n", type=int, default=30)
    s = sp.add_parser("speed"); s.add_argument("--data", required=True); s.add_argument("--out", default="results"); s.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "eval": cmd_eval, "speed": cmd_speed, "finetune": cmd_finetune, "speed-e2e": cmd_speed_e2e}[a.cmd](a)
