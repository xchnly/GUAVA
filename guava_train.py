"""
guava_train.py — Training & evaluasi bersama.

EVALUATOR TERPADU (semua family dinilai dengan kode yang sama, ruang letterbox C.EVAL_SIZE):
  * mAP50 / mAP50-95 (box & mask), COCO 101-point, rata-rata per kelas yang punya GT.
  * precision / recall (mask, IoU 0.5) pada ambang skor dengan F1 maksimum.
  * Model semantik → instance via connected components per kelas (skor = mean prob komponen).
  * YOLO → instance langsung dari ultralytics predict (retina_masks). Metrik native
    ultralytics.val juga dicatat (kolom ultra_*).
"""
import copy
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import guava_data as D
import guava_models as M
from guava_core import C, count_flops, count_params, file_mb, log, measure_latency

IOU_THR = np.linspace(0.5, 0.95, 10)


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16 if C.AMP_DTYPE == "bf16" else torch.float16)
    return torch.autocast("cpu", enabled=False)


# =============================================================================
# EVALUATOR
# =============================================================================
def _boxes_from_masks(masks):
    out = np.zeros((len(masks), 4), np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(xs):
            out[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return out


def _box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _mask_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ta = torch.from_numpy(a.reshape(len(a), -1)).float()
    tb = torch.from_numpy(b.reshape(len(b), -1)).float()
    inter = ta @ tb.T
    union = ta.sum(1)[:, None] + tb.sum(1)[None] - inter
    return (inter / union.clamp(min=1)).numpy()


def _ap101(rec, prec):
    """AP COCO 101-titik: presisi terinterpolasi = max presisi pada recall ≥ r."""
    env = np.flip(np.maximum.accumulate(np.flip(prec)))
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))


def compute_metrics(preds, gts, nc):
    """preds/gts: list per gambar. pred = {masks (N,E,E) bool, boxes (N,4), scores, cls};
    gt = {masks, cls}. Return dict mAP50_box, mAP50-95_box, mAP50_mask, mAP50-95_mask, precision, recall."""
    out = {}
    for typ in ("mask", "box"):
        stats = {c: {"scores": [], "tp": [], "n_gt": 0} for c in range(nc)}
        for p, g in zip(preds, gts):
            gb = _boxes_from_masks(g["masks"])
            if typ == "mask":
                iou = _mask_iou(p["masks"], g["masks"])
            else:
                iou = _box_iou(p["boxes"], gb)
            for c in range(nc):
                gi = np.where(g["cls"] == c)[0]
                pi = np.where(p["cls"] == c)[0]
                stats[c]["n_gt"] += len(gi)
                if not len(pi):
                    continue
                pi = pi[np.argsort(-p["scores"][pi])]
                tp = np.zeros((len(pi), len(IOU_THR)), bool)
                if len(gi):
                    sub = iou[np.ix_(pi, gi)]
                    for t, thr in enumerate(IOU_THR):
                        used = np.zeros(len(gi), bool)
                        for j in range(len(pi)):
                            cand = np.where((~used) & (sub[j] >= thr))[0]
                            if len(cand):
                                k = cand[np.argmax(sub[j, cand])]
                                used[k] = True
                                tp[j, t] = True
                stats[c]["scores"].append(p["scores"][pi])
                stats[c]["tp"].append(tp)
        aps, precs, recs = [], [], []
        for c, s in stats.items():
            if s["n_gt"] == 0:
                continue
            if not s["scores"]:
                aps.append(np.zeros(len(IOU_THR)))
                precs.append(0.0)
                recs.append(0.0)
                continue
            sc = np.concatenate(s["scores"])
            tp = np.concatenate(s["tp"])[np.argsort(-sc)]
            ctp = np.cumsum(tp, 0)
            cfp = np.cumsum(~tp, 0)
            rec = ctp / s["n_gt"]
            prec = ctp / np.maximum(ctp + cfp, 1)
            aps.append(np.array([_ap101(rec[:, t], prec[:, t]) for t in range(len(IOU_THR))]))
            f1 = 2 * prec[:, 0] * rec[:, 0] / np.maximum(prec[:, 0] + rec[:, 0], 1e-9)
            b = int(np.argmax(f1))
            precs.append(float(prec[b, 0]))
            recs.append(float(rec[b, 0]))
        aps = np.array(aps) if aps else np.zeros((1, len(IOU_THR)))
        out[f"mAP50_{typ}"] = float(aps[:, 0].mean())
        out[f"mAP50-95_{typ}"] = float(aps.mean())
        if typ == "mask":
            out["precision"] = float(np.mean(precs)) if precs else 0.0
            out["recall"] = float(np.mean(recs)) if recs else 0.0
    return out


def semantic_to_instances(prob):
    """prob: torch (K,E,E) softmax. Return dict pred instance (connected components per kelas)."""
    K = prob.shape[0]
    lab = prob.argmax(0).cpu().numpy().astype(np.uint8)
    pn = prob.float().cpu().numpy()
    masks, scores, cls, boxes = [], [], [], []
    for c in range(1, K):
        n, cc, st, _ = cv2.connectedComponentsWithStats((lab == c).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < C.EVAL_MIN_AREA:
                continue
            m = cc == k
            masks.append(m)
            scores.append(float(pn[c][m].mean()))
            cls.append(c - 1)
            x, y, w, h = st[k, :4]
            boxes.append([x, y, x + w, y + h])
    return _pack(masks, scores, cls, boxes, prob.shape[-1])


def _pack(masks, scores, cls, boxes, E):
    if not masks:
        return {"masks": np.zeros((0, E, E), bool), "scores": np.zeros(0, np.float32),
                "cls": np.zeros(0, np.int64), "boxes": np.zeros((0, 4), np.float32)}
    o = np.argsort(-np.asarray(scores))[:C.EVAL_MAX_DET]
    return {"masks": np.stack(masks)[o], "scores": np.asarray(scores, np.float32)[o],
            "cls": np.asarray(cls, np.int64)[o], "boxes": np.asarray(boxes, np.float32)[o]}


@torch.no_grad()
def predict_semantic(model_or_fn, items, S, batch=8):
    """model_or_fn: nn.Module (B,3,S,S)→logits, atau callable(np uint8 batch NCHW)→logits np/torch."""
    E = C.EVAL_SIZE
    loader = D.make_loader(items, S, batch, train=False, workers=min(4, C.NUM_WORKERS))
    is_mod = isinstance(model_or_fn, nn.Module)
    if is_mod:
        model_or_fn.eval()
        d = module_device(model_or_fn)
    preds = []
    for b in loader:
        if is_mod:
            with amp_ctx() if d.type == "cuda" else torch.autocast("cpu", enabled=False):
                logits = model_or_fn(b["image"].to(d).float())
        else:
            logits = torch.as_tensor(model_or_fn(b["image"].numpy()))
        prob = torch.softmax(logits.float(), 1)
        prob = F.interpolate(prob, size=(E, E), mode="bilinear", align_corners=False)
        for p in prob:
            preds.append(semantic_to_instances(p))
    return preds


def eval_semantic(model_or_fn, items, S, gts=None, batch=8):
    gts = gts if gts is not None else D.eval_gt(items)
    nc = model_or_fn.nc if isinstance(model_or_fn, nn.Module) else getattr(model_or_fn, "nc", None)
    preds = predict_semantic(model_or_fn, items, S, batch)
    return compute_metrics(preds, gts, nc)


def module_device(m):
    for t in list(m.parameters()) + list(m.buffers()):
        return t.device
    return torch.device("cpu")


def prefix(d, p):
    return {f"{p}_{k}": v for k, v in d.items()}


# =============================================================================
# LOSS
# =============================================================================
def dice_loss(logits, target, eps=1.0):
    K = logits.shape[1]
    p = torch.softmax(logits.float(), 1)
    oh = F.one_hot(target, K).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def task_loss(logits, target):
    return F.cross_entropy(logits.float(), target) + dice_loss(logits, target)


# =============================================================================
# KNOWLEDGE DISTILLATION
# =============================================================================
class KDTeacher:
    """Teacher (YOLO atau SegNet) → target lunak (B,K,S,S) + feature map.
    YOLO (instance) dikonversi ke peta probabilitas per kelas: p_c = max(conf × mask),
    background = 1 − max_c p_c; lalu dinormalisasi. 'Logit' teacher = log(p)."""

    def __init__(self, meta, path, nc):
        self.meta, self.nc, self.kind = meta, nc, meta["kind"]
        self._feat = None
        self._hooked = False
        d = dev()
        if self.kind == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(str(path))
        else:
            self.model, _ = M.load_model(path, d)
            self.model.capture_feat = True
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _hook_yolo(self):
        try:
            head = self.model.predictor.model.model.model[-1]
            head.register_forward_pre_hook(lambda m, inp: setattr(self, "_feat", inp[0][0]))
            self._hooked = True
        except Exception as e:
            log.warning(f"[KD] gagal hook fitur YOLO ({e}); feature-KD dimatikan")
            self._hooked = None

    @torch.no_grad()
    def __call__(self, x):
        """x: float (B,3,S,S) 0..255 di device. Return (teacher_logits, teacher_feat|None)."""
        B, _, S, _ = x.shape
        if self.kind != "yolo":
            with amp_ctx():
                logits = self.model(x)
            return logits.float(), (self.model.feat.float() if self.model.feat is not None else None)
        imgs = x.byte().permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]  # RGB→BGR
        K = self.nc + 1
        probs, feats = [], []
        for i in range(0, B, C.KD_TEACHER_CHUNK):
            chunk = [np.ascontiguousarray(im) for im in imgs[i:i + C.KD_TEACHER_CHUNK]]
            res = self.model.predict(chunk, imgsz=S, conf=C.KD_TEACHER_CONF, retina_masks=True, max_det=50,
                                     verbose=False, half=torch.cuda.is_available(), device=x.device)
            if self._hooked is False:
                self._hook_yolo()
                res = self.model.predict(chunk, imgsz=S, conf=C.KD_TEACHER_CONF, retina_masks=True, max_det=50,
                                         verbose=False, half=torch.cuda.is_available(), device=x.device)
            if self._hooked and self._feat is not None:
                feats.append(self._feat.float().clone())
            for r in res:
                P = torch.zeros(K, S, S, device=x.device)
                if r.masks is not None and len(r.masks):
                    m = r.masks.data.float().clone()
                    if m.shape[-2:] != (S, S):
                        m = F.interpolate(m[None], size=(S, S), mode="nearest")[0]
                    conf = r.boxes.conf.float().clone()
                    cls = r.boxes.cls.long().clone()
                    for j in range(len(m)):
                        c = int(cls[j]) + 1
                        P[c] = torch.maximum(P[c], conf[j] * m[j])
                P[0] = 1 - P[1:].max(0).values
                P = P.clamp(min=1e-4)
                probs.append(P / P.sum(0, keepdim=True))
        feat = torch.cat(feats) if feats and sum(len(f) for f in feats) == B else None
        return torch.log(torch.stack(probs)), feat


class KDLoss(nn.Module):
    """L = alpha·L_task + beta·L_kd_logits + gamma·L_kd_features
       L_kd_logits  = KL(softmax(t/T) || softmax(s/T)) · T²
       L_kd_features= MSE(norm(proj(f_s)), norm(f_t))   (proj 1×1 conv bila dim beda)"""

    def __init__(self, teacher, alpha, T, beta, gamma):
        super().__init__()
        self.teacher, self.alpha, self.T, self.beta, self.gamma = teacher, alpha, T, beta, gamma
        self.proj = None

    def setup(self, student, S):
        d = dev()
        x = torch.zeros(2, 3, S, S, device=d)
        student.capture_feat = True
        with torch.no_grad():
            student(x)
            _, tf = self.teacher(x)
        sf = student.feat
        if self.gamma > 0 and tf is not None and sf is not None:
            self.proj = nn.Conv2d(sf.shape[1], tf.shape[1], 1, bias=False).to(d)
        else:
            if self.gamma > 0:
                log.warning("[KD] feature map tidak tersedia → gamma=0")
            self.gamma = 0.0
        return self

    def forward(self, student, x, s_logits, target):
        t_logits, t_feat = self.teacher(x)
        T = self.T
        l_task = task_loss(s_logits, target)
        l_kd = F.kl_div(F.log_softmax(s_logits.float() / T, 1), F.softmax(t_logits / T, 1),
                        reduction="batchmean") * T * T / (x.shape[-1] * x.shape[-2])
        l_feat = torch.zeros((), device=x.device)
        if self.gamma > 0 and t_feat is not None and student.feat is not None:
            sf = self.proj(student.feat.float())
            if sf.shape[-2:] != t_feat.shape[-2:]:
                sf = F.interpolate(sf, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            l_feat = F.mse_loss(F.normalize(sf, dim=1), F.normalize(t_feat.float(), dim=1)) * sf.shape[1]
        loss = self.alpha * l_task + self.beta * l_kd + self.gamma * l_feat
        return loss, {"l_task": float(l_task), "l_kd": float(l_kd), "l_feat": float(l_feat)}


# =============================================================================
# TRAINING model semantik (Transformer / CNN / Edge / KD / fine-tune pruning / QAT)
# =============================================================================
def train_semantic(model, fd, S, epochs, batch, lr, seed, save_path=None, kd=None, patience=None,
                   history=None, meta=None, save_full=False, device=None, amp=True):
    """Simpan bobot HANYA saat val mAP50-95_mask naik (1 file, ditimpa). Return info dict.
    Bobot terbaik dimuat kembali ke `model` di akhir."""
    d = device or dev()
    model.to(d)
    patience = patience or C.PATIENCE
    tl = D.make_loader(fd.train, S, batch, train=True, seed=seed)
    gts_val = D.eval_gt(fd.val)
    params = [p for p in model.parameters() if p.requires_grad]
    if kd is not None:
        kd.setup(model, S)
        if kd.proj is not None:
            params += list(kd.proj.parameters())
    model.capture_feat = kd is not None
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    steps = max(1, len(tl)) * epochs
    warm = min(steps // 5, max(1, len(tl)) * C.WARMUP_EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    best, best_ep, best_state = -1.0, -1, None
    ep = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n, parts = time.time(), 0.0, 0, {}
        for b in tl:
            x = b["image"].to(d, non_blocking=True).float()
            y = b["sem"].to(d, non_blocking=True)
            with amp_ctx() if amp else torch.autocast(d.type, enabled=False):
                logits = model(x)
            if kd is not None:
                loss, pr = kd(model, x, logits, y)
                for k, v in pr.items():
                    parts[k] = parts.get(k, 0) + v
            else:
                loss = task_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += float(loss)
            n += 1
        row = {"epoch": ep, "train_loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"],
               "epoch_sec": round(time.time() - t0, 1), **{k: v / max(n, 1) for k, v in parts.items()}}
        if ep % C.VAL_EVERY == 0 or ep == epochs:
            vm = eval_semantic(model, fd.val, S, gts_val, batch=max(1, min(batch, 16)))
            row.update(prefix(vm, "val"))
            fit = vm["mAP50-95_mask"] + 1e-3 * vm["mAP50_mask"]
            if fit > best:
                best, best_ep = fit, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path:
                    M.save_model(model, save_path, meta={**(meta or {}), "epoch": ep, "val_fit": fit},
                                 full=save_full)
        if history is not None:
            history(row)
        if best_ep > 0 and ep - best_ep >= patience:
            log.info(f"[TRAIN] early stop ep {ep} (best {best_ep})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.capture_feat = False
    return {"best_epoch": best_ep, "epochs_run": ep, "imgsz": S, "batch": batch}


def with_oom_retry(fn, batch, min_batch=1):
    """Jalankan fn(batch); kalau CUDA OOM → batch /2 dan ulangi."""
    while True:
        try:
            return fn(batch), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= min_batch:
                raise
            batch = max(min_batch, batch // 2)
            log.warning(f"[OOM] turunkan batch → {batch}")


def profile_semseg(model, S, weight_path=None):
    """params_M, flops_G, size_MB, latency CPU/GPU (batch=1)."""
    m = copy.deepcopy(model).float().cpu().eval()
    m.capture_feat = False
    out = {"params_M": round(count_params(m), 3), "flops_G": round(count_flops(m, S), 3),
           "size_MB": round(file_mb(weight_path), 2) if weight_path else float("nan"),
           "latency_cpu_ms": round(measure_latency(m, S, "cpu"), 2)}
    out["latency_gpu_ms"] = round(measure_latency(m, S, "cuda", fp16=True), 2) if torch.cuda.is_available() \
        else float("nan")
    out["latency_ms"] = out["latency_cpu_ms"]
    del m
    return out


def evaluate_full(model, fd, S):
    r = {}
    r.update(prefix(eval_semantic(model, fd.val, S), "val"))
    r.update(prefix(eval_semantic(model, fd.test, S), "test"))
    return r


# =============================================================================
# YOLO (ultralytics)
# =============================================================================
class GuavaAlbu:
    """Pengganti ultralytics Albumentations: augmentasi piksel baru (brightness/contrast, HSV,
    MotionBlur, CoarseDropout p=0.2). Augmentasi spasial YOLO lewat hyp: flip lr/ud, degrees=20,
    scale=0.2 (0.8–1.2), shear=5; RandomRotate90 dipakai di sampel balancing offline."""

    def __init__(self, p=1.0, transforms=None, flip_idx=None, **kw):
        self.p = p
        self.contains_spatial = False
        try:
            import albumentations as A
            self.transform = A.Compose(D.pixel_transforms(A))
        except Exception:
            self.transform = None

    def __call__(self, labels):
        if self.transform is not None and np.random.rand() < self.p:
            labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


def yolo_pretrained(name):
    pdir = C.WEIGHTS_DIR / "pretrained"
    pdir.mkdir(parents=True, exist_ok=True)
    fn = M.REGISTRY[name][2]["weights"]
    if M.NO_PRETRAINED:
        return Path(fn.replace(".pt", ".yaml"))   # uji offline: arsitektur tanpa bobot
    dst = pdir / fn
    if not dst.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        got = Path(attempt_download_asset(str(dst)))
        if got.exists() and got.resolve() != dst.resolve():
            got.replace(dst)
    if not dst.exists():
        raise FileNotFoundError(f"gagal download {fn}")
    return dst


def train_yolo(name, fd, S, epochs, batch, seed, run_name, history=None):
    import os
    import pandas as pd
    import ultralytics.data.augment as UA
    from ultralytics import YOLO
    UA.Albumentations = GuavaAlbu
    data = D.materialize_yolo(fd)
    model = YOLO(str(yolo_pretrained(name)))
    wdir = C.RUNS_DIR / run_name / "weights"

    def drop_last(trainer):  # aturan #7: hapus last.pt / epoch*.pt segera setelah disimpan
        for p in list(Path(trainer.save_dir, "weights").glob("*.pt")):
            if p.name != "best.pt":
                p.unlink(missing_ok=True)

    model.add_callback("on_train_epoch_end", drop_last)
    model.add_callback("on_fit_epoch_end", drop_last)
    cwd = os.getcwd()
    os.chdir(C.TMP_DIR)  # file nyasar (yolo11n.pt dari AMP check, font) masuk tmp/
    try:
        model.train(data=str(data), epochs=epochs, imgsz=S, batch=batch, seed=seed, deterministic=False,
                    project=str(C.RUNS_DIR), name=run_name, exist_ok=True, patience=C.PATIENCE, plots=True,
                    save_period=-1, workers=C.NUM_WORKERS, fliplr=0.5, flipud=0.5, degrees=20.0, scale=0.2,
                    shear=5.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, mosaic=1.0, close_mosaic=10, amp=True,
                    cache=False, verbose=False, val=True, device=0 if torch.cuda.is_available() else "cpu")
    finally:
        os.chdir(cwd)
    csv = C.RUNS_DIR / run_name / "results.csv"
    if history is not None and csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        for r in df.to_dict("records"):
            history(r)
    best = wdir / "best.pt"
    if not best.exists():
        raise FileNotFoundError("best.pt tidak terbentuk")
    return best


@torch.no_grad()
def predict_yolo(yolo, items, S, bs=16):
    E = C.EVAL_SIZE
    preds = []
    for i in range(0, len(items), bs):
        imgs = [np.ascontiguousarray(D.letterbox(D.load_image(it), S)[..., ::-1]) for it in items[i:i + bs]]
        res = yolo.predict(imgs, imgsz=S, conf=0.001, iou=0.7, retina_masks=True, max_det=C.EVAL_MAX_DET,
                           verbose=False, half=torch.cuda.is_available())
        for r in res:
            if r.masks is None or not len(r.masks):
                preds.append(_pack([], [], [], [], E))
                continue
            m = F.interpolate(r.masks.data.float()[None], size=(E, E), mode="nearest")[0] > 0.5
            preds.append({"masks": m.cpu().numpy(), "scores": r.boxes.conf.float().cpu().numpy(),
                          "cls": r.boxes.cls.long().cpu().numpy(),
                          "boxes": (r.boxes.xyxy.float().cpu().numpy() * E / S).astype(np.float32)})
    return preds


def eval_yolo(best, fd, S, native=True):
    from ultralytics import YOLO
    yolo = YOLO(str(best))
    out = {}
    for split, its in (("val", fd.val), ("test", fd.test)):
        out.update(prefix(compute_metrics(predict_yolo(yolo, its, S), D.eval_gt(its), fd.nc), split))
    if native:
        data = D.materialize_yolo(fd)
        for split in ("val", "test"):
            try:
                mt = yolo.val(data=str(data), split=split, imgsz=S, batch=C.BATCH_YOLO, plots=False, verbose=False,
                              project=str(C.RUNS_DIR), name="_val", exist_ok=True)
                out.update({f"ultra_{split}_mAP50_box": mt.box.map50, f"ultra_{split}_mAP50-95_box": mt.box.map,
                            f"ultra_{split}_mAP50_mask": mt.seg.map50, f"ultra_{split}_mAP50-95_mask": mt.seg.map})
            except Exception as e:
                log.warning(f"[YOLO] native val gagal: {e}")
    net = yolo.model.float().eval()
    out.update({"params_M": round(count_params(net), 3), "flops_G": round(count_flops(net, S), 3),
                "size_MB": round(file_mb(best), 2),
                "latency_cpu_ms": round(measure_latency(net, S, "cpu"), 2),
                "latency_gpu_ms": round(measure_latency(net, S, "cuda", fp16=True), 2)
                if torch.cuda.is_available() else float("nan")})
    out["latency_ms"] = out["latency_cpu_ms"]
    del yolo
    return out


# =============================================================================
# HELPER unit semantik (dipakai Part 2, 3, 4, 5)
# =============================================================================
def semantic_unit(book, key, base, attempt, fd, S, epochs, batch, lr, seed, save_path,
                  make_model, make_kd=None, save_full=False, eval_test=True):
    """Train (dengan OOM retry) → eval val+test → profil. Return (model, row_dict)."""
    from guava_core import seed_everything
    seed_everything(seed)
    hist_base = {k: base[k] for k in ("part", "model", "seed", "fold", "technique", "hyperparam") if k in base}

    def history(r):
        book.add("history", {"unit": key, "attempt": attempt, **hist_base, **r}, flush=False)

    holder = {}

    def go(b):
        holder.clear()
        model = make_model()
        holder["model"] = model
        kd = make_kd() if make_kd else None
        return train_semantic(model, fd, S, epochs, b, lr, seed, save_path=save_path, kd=kd,
                              history=history, meta={"imgsz": S, **base}, save_full=save_full)

    info, used = with_oom_retry(go, batch)
    model = holder["model"]
    res = {"best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"], "batch": used}
    res.update(prefix(eval_semantic(model, fd.val, S), "val"))
    if eval_test:
        res.update(prefix(eval_semantic(model, fd.test, S), "test"))
    res.update(profile_semseg(model, S, save_path))
    return model, res


def baseline_part(part, xlsx_name, groups, args, keep_work, teacher_candidate=True):
    """Runner generik Part 2 & 3. groups: list (family, [models], epochs, batch, lr).
    keep_work=True → bobot tetap di weights/work/s_f/ (dipakai Part 5); False → dihapus setelah
    dipromosikan ke weights/best (+ kandidat teacher)."""
    from guava_core import (DG, PartState, ResultBook, env_report, ensure_deps, pick_imgsz, promote_final,
                            run_unit, unit_key, update_teacher, work_dir, write_part_done)
    ensure_deps()
    env_report()
    DG.report_disk()
    DG.guard_before_training()
    master = D.load_master()
    nc = master["nc"]
    plan = []
    for fam, names, epochs, batch, lr in groups:
        ok = M.usable(names, nc, family=fam)
        if args.model_list:
            ok = [n for n in ok if n in args.model_list]
        plan += [(fam, n, epochs, batch, lr) for n in ok]
    tokens = sum((M.cache_tokens(p[1]) for p in plan), [])
    state = PartState(part)
    book = ResultBook(C.EXCEL_DIR / f"{xlsx_name}.xlsx")
    expected = [unit_key(p[1], s, f) for s in C.SEEDS for f in C.FOLDS for p in plan]
    for seed in args.seeds:
        for fold in args.folds:
            if all(state.is_done(unit_key(p[1], seed, fold)) for p in plan):
                continue
            fd = D.prepare_fold(seed, fold, master)
            for fam, name, epochs, batch, lr in plan:
                key = unit_key(name, seed, fold)
                if state.is_done(key):
                    continue
                S = pick_imgsz(seed, fold, name)
                base = {"part": part, "model": name, "family": fam, "seed": seed, "fold": fold, "imgsz": S,
                        "technique": "baseline", "hyperparam": "-"}
                wpath = work_dir(seed, fold) / f"{part}_{name}.pt"   # prefix part: model sama di Part 2 & 3

                def fn(attempt, name=name, fam=fam, S=S, key=key, base=base, wpath=wpath, epochs=epochs,
                       batch=batch, lr=lr, fd=fd, seed=seed, fold=fold):
                    model, res = semantic_unit(book, key, base, attempt, fd, S, epochs, batch, lr, seed, wpath,
                                               make_model=lambda: M.build(name, nc, pretrained=True))
                    promote_final(wpath, fam, name, seed, fold, res["val_mAP50-95_mask"], {"imgsz": S})
                    if teacher_candidate:
                        update_teacher(seed, fold, name, "semantic", fam, res[C.TEACHER_SELECT_METRIC], wpath,
                                       S, part)
                    if not keep_work:
                        wpath.unlink(missing_ok=True)
                    del model
                    return res

                run_unit(state, book, key, base, fn, keep_models=tokens, keep_folds=[fd.name])
                DG.clean_all(keep_models=tokens, keep_folds=[fd.name], verbose=False)
            book.flush(include_history=True)
    book.flush(include_history=True)
    if not args.no_final:
        write_part_done(part, state, expected, {"models": [p[1] for p in plan]})
        free = DG.disk_free_gb()
        DG.clean_all(keep_models=[] if free < C.CACHE_PURGE_FREE_GB else tokens, keep_folds=[])
        DG.report_disk()
    return plan
