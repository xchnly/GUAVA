"""
guava_data.py — Dataset guava (instance segmentation, polygon YOLO-seg):
  NDJSON → cache master (.npy uint8) → split 80/10/10 × 5-fold × 5 seed (stratified)
  → balancing offline (augmentasi baru) → Dataset PyTorch / dataset YOLO sementara.

PREPROCESSING BARU (beda dari notebook lama 85/10/5):
  * Multi-resolusi: imgsz ∈ C.IMGSZ_CHOICES, dipilih per run (lihat guava_core.pick_imgsz).
  * Augmentasi train: HorizontalFlip, VerticalFlip, RandomRotate90, Affine(scale 0.8–1.2,
    rotate ±20, shear ±5), RandomBrightnessContrast, HueSaturationValue, MotionBlur, CoarseDropout(p=0.2).
  * Val/Test: letterbox saja.
  * Balancing train: target C.BALANCE gambar/kelas (kelas dominan), maksimal +C.BALANCE_MAX_AUG.
"""
import json
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from guava_core import C, fold_name, log, read_json, stable_rng, write_json

MASTER = C.CACHE_DIR / "master"
PAD_VALUE = 114
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# =============================================================================
# NDJSON
# =============================================================================
def _parse_segments(ann, w, h):
    """annotations = {"segments": [[cls, x1, y1, ...], ...]} (format Ultralytics NDJSON).
    Mendukung juga 'boxes' [cls, xc, yc, bw, bh] (dikonversi ke polygon persegi)."""
    if not ann:
        return []
    if isinstance(ann, list):
        ann = {"segments": ann}
    key = "segments" if "segments" in ann else next(iter(ann))
    out = []
    for item in ann.get(key) or []:
        if isinstance(item, dict):
            cls = int(item.get("class", item.get("class_id", item.get("category_id", 0))))
            pts = np.asarray(item.get("points", item.get("segmentation", [])), dtype=np.float32).reshape(-1)
        else:
            if len(item) < 5:
                continue
            cls, pts = int(item[0]), np.asarray(item[1:], dtype=np.float32)
        if key == "boxes" and len(pts) == 4:
            xc, yc, bw, bh = pts
            pts = np.array([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc - bh / 2,
                            xc + bw / 2, yc + bh / 2, xc - bw / 2, yc + bh / 2], dtype=np.float32)
        if len(pts) < 6 or len(pts) % 2:
            continue
        pts = pts.reshape(-1, 2)
        if pts.max() > 1.5 and w and h:  # koordinat pixel → normalisasi
            pts = pts / np.array([w, h], dtype=np.float32)
        out.append([cls, np.clip(pts, 0, 1).round(6).reshape(-1).tolist()])
    return out


def load_ndjson(path=None):
    path = Path(path or C.NDJSON_PATH)
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    head = lines[0] if lines and lines[0].get("type") == "dataset" else {}
    recs = [r for r in lines if r.get("type", "image") == "image" and ("file" in r or "url" in r)]
    names = {int(k): v for k, v in (head.get("class_names") or head.get("names") or {}).items()}
    out = []
    for i, r in enumerate(recs):
        segs = _parse_segments(r.get("annotations", {}), r.get("width"), r.get("height"))
        out.append({"i": i, "file": r.get("file") or Path(str(r.get("url"))).name.split("?")[0],
                    "url": r.get("url"), "split": r.get("split", "train"), "segs": segs,
                    "local_root": head.get("path")})
    ids = {s[0] for r in out for s in r["segs"]}
    nc = max(list(names) + list(ids)) + 1 if (names or ids) else 1
    names = {i: names.get(i, f"class{i}") for i in range(nc)}
    log.info(f"[DATA] NDJSON {path.name}: {len(out)} gambar, nc={nc}, names={names}")
    return {"names": names, "nc": nc}, out


def load_yolo_dir(root):
    """Fallback: dataset YOLO-seg biasa (images/*, labels/*). Set GUAVA_YOLO_DIR."""
    root = Path(root)
    names = {}
    for y in list(root.glob("*.yaml")):
        try:
            import yaml
            d = yaml.safe_load(y.read_text())
            n = d.get("names", {})
            names = dict(enumerate(n)) if isinstance(n, list) else {int(k): v for k, v in n.items()}
            break
        except Exception:
            pass
    out = []
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts):
        lab = Path(str(img).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
        segs = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                v = line.split()
                if len(v) >= 7:
                    segs.append([int(v[0]), [float(x) for x in v[1:]]])
        out.append({"i": len(out), "file": str(img), "url": None, "split": "train", "segs": segs,
                    "local_root": None})
    ids = {s[0] for r in out for s in r["segs"]}
    nc = max(list(names) + list(ids)) + 1 if (names or ids) else 1
    return {"names": {i: names.get(i, f"class{i}") for i in range(nc)}, "nc": nc}, out


# =============================================================================
# MASTER CACHE (.npy uint8)
# =============================================================================
def _find_local(rec):
    f = Path(rec["file"])
    if f.is_absolute() and f.exists():
        return f
    base = C.NDJSON_PATH.parent
    cands = []
    if rec.get("local_root"):
        cands.append(base / rec["local_root"] / "images" / rec["split"] / f.name)
    cands += [C.RAW_DIR / rec["split"] / f.name, C.RAW_DIR / f.name, C.RAW_DIR / "images" / rec["split"] / f.name,
              base / "images" / rec["split"] / f.name, base / "images" / f.name, base / f.name,
              C.RAW_DIR / f"{rec['i']:06d}_{f.name}"]
    return next((c for c in cands if c.exists()), None)


def _download(rec):
    dst = C.RAW_DIR / f"{rec['i']:06d}_{Path(rec['file']).name}"
    if dst.exists():
        return dst
    url = rec.get("url")
    if not url or not str(url).startswith("http"):
        return None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return dst
        except Exception as e:
            if attempt == 3:
                log.warning(f"[DATA] gagal download {url}: {e}")
            else:
                import time
                time.sleep(2 ** attempt)
    return None


def _encode(img_rgb):
    if C.CACHE_FORMAT == "raw":
        return img_rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, C.CACHE_JPEG_QUALITY])
    return buf.reshape(-1)


def decode(arr):
    """npy uint8 → RGB uint8 HxWx3 (npy berisi byte JPEG atau array mentah)."""
    if arr.ndim == 1:
        return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return arr


def load_image(item):
    return decode(np.load(item["src"]))


def _dominant(segs, global_counts, nc):
    if not segs:
        return nc  # stratum "tanpa objek"
    cnt = {}
    for c, _ in segs:
        cnt[c] = cnt.get(c, 0) + 1
    top = max(cnt.values())
    return min((c for c in cnt if cnt[c] == top), key=lambda c: global_counts.get(c, 0))


def build_master_cache():
    """Idempotent & resumable. Setelah selesai .jpg mentah dihapus (clean_raw_jpg)."""
    idx_path = MASTER / "index.json"
    if idx_path.exists():
        return read_json(idx_path)
    MASTER.mkdir(parents=True, exist_ok=True)
    yolo_dir = os.environ.get("GUAVA_YOLO_DIR")
    meta, recs = load_yolo_dir(yolo_dir) if yolo_dir else load_ndjson()

    def work(rec):
        out = MASTER / f"{rec['i']:06d}.npy"
        if out.exists():
            arr = decode(np.load(out))
            return rec, arr.shape[0], arr.shape[1]
        p = _find_local(rec) or _download(rec)
        if p is None:
            return rec, None, None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return rec, None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = C.MASTER_MAX_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        np.save(out, _encode(img))
        return rec, img.shape[0], img.shape[1]

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(work, recs))
    gcount = {}
    for rec, h, _ in results:
        if h:
            for c, _ in rec["segs"]:
                gcount[c] = gcount.get(c, 0) + 1
    items = []
    for rec, h, w in results:
        if not h:
            continue
        items.append({"id": rec["i"], "src": str(MASTER / f"{rec['i']:06d}.npy"), "file": rec["file"],
                      "h": h, "w": w, "segs": rec["segs"], "dom": _dominant(rec["segs"], gcount, meta["nc"])})
    missing = len(recs) - len(items)
    data = {"names": {str(k): v for k, v in meta["names"].items()}, "nc": meta["nc"], "items": items,
            "missing": missing, "format": C.CACHE_FORMAT}
    write_json(idx_path, data)
    log.info(f"[DATA] cache master: {len(items)} gambar ({missing} gagal), "
             f"{sum(os.path.getsize(i['src']) for i in items)/1e6:.1f} MB")
    return data


def load_master():
    d = read_json(MASTER / "index.json")
    if d is None:
        d = build_master_cache()
    d["names"] = {int(k): v for k, v in d["names"].items()}
    return d


# =============================================================================
# SPLIT 80/10/10 × 5-fold (stratified by kelas dominan)
# =============================================================================
def make_split(seed, fold, items):
    """Per seed: StratifiedKFold(10) → 10 chunk. Fold k: test=chunk 2k, val=chunk 2k+1, train=sisa.
    → 80/10/10 persis, stratified, dan tiap gambar tepat sekali di val ATAU test per seed."""
    from sklearn.model_selection import KFold, StratifiedKFold
    import warnings
    y = np.array([it["dom"] for it in items])
    n = len(items)
    k = min(10, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chunks = [te for _, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y)]
        except ValueError:
            chunks = [te for _, te in KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))]
    te, va = chunks[(2 * fold) % k], chunks[(2 * fold + 1) % k]
    tr = np.setdiff1d(np.arange(n), np.concatenate([te, va]))
    return {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}


# =============================================================================
# AUGMENTASI (albumentations 2.x; fallback argumen 1.x)
# =============================================================================
def _A():
    import albumentations as A
    return A


def _affine(A, p):
    try:
        return A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT,
                        fill=PAD_VALUE, fill_mask=0, p=p)
    except TypeError:
        return A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), shear=(-5, 5), mode=cv2.BORDER_CONSTANT,
                        cval=PAD_VALUE, cval_mask=0, p=p)


def _dropout(A, p=0.2):
    try:
        return A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.03, 0.12),
                               hole_width_range=(0.03, 0.12), fill=PAD_VALUE, p=p)
    except TypeError:
        return A.CoarseDropout(max_holes=8, max_height=0.12, max_width=0.12, min_holes=1,
                               fill_value=PAD_VALUE, p=p)


def pixel_transforms(A):
    return [A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 15, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            _dropout(A, 0.2)]


def train_aug():
    A = _A()
    return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                      _affine(A, 0.5), *pixel_transforms(A)])


def strong_aug():
    """Untuk sampel balancing offline: spatial selalu aktif agar tidak duplikat identik."""
    A = _A()
    return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.75),
                      _affine(A, 0.9), *pixel_transforms(A)])


def seg_masks(segs, h, w):
    masks, cls = [], []
    for c, pts in segs:
        m = np.zeros((h, w), np.uint8)
        p = (np.asarray(pts, np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        cv2.fillPoly(m, [p], 1)
        if m.any():
            masks.append(m)
            cls.append(int(c))
    return masks, cls


def apply_aug(tf, img, masks):
    if masks:
        r = tf(image=img, masks=list(masks))
        return r["image"], [np.asarray(m) for m in r["masks"]]
    return tf(image=img)["image"], []


def masks_to_segs(masks, cls, min_area=10):
    segs = []
    for m, c in zip(masks, cls):
        h, w = m.shape
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [x for x in cnts if cv2.contourArea(x) >= min_area and len(x) >= 3]
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) / [w, h]
        segs.append([int(c), np.clip(cnt, 0, 1).round(6).reshape(-1).tolist()])
    return segs


# =============================================================================
# FOLD (split + balancing offline) → cache/s{seed}_f{fold}/
# =============================================================================
class FoldData:
    def __init__(self, seed, fold, master, split, extras):
        self.seed, self.fold = seed, fold
        self.names, self.nc = master["names"], master["nc"]
        items = master["items"]
        self.train = [items[i] for i in split["train"]] + extras
        self.train_orig = [items[i] for i in split["train"]]
        self.val = [items[i] for i in split["val"]]
        self.test = [items[i] for i in split["test"]]
        self.extras = extras

    @property
    def name(self):
        return fold_name(self.seed, self.fold)


def prepare_fold(seed, fold, master=None):
    master = master or load_master()
    fdir = C.CACHE_DIR / fold_name(seed, fold)
    meta_p = fdir / "fold.json"
    meta = read_json(meta_p)
    if meta is None:
        split = make_split(seed, fold, master["items"])
        extras = _balance(seed, fold, master, split, fdir)
        meta = {"split": split, "extras": extras}
        write_json(meta_p, meta)
    return FoldData(seed, fold, master, meta["split"], meta["extras"])


def _balance(seed, fold, master, split, fdir):
    items, nc = master["items"], master["nc"]
    train = [items[i] for i in split["train"]]
    by_cls = {c: [it for it in train if it["dom"] == c] for c in range(nc)}
    deficit = {c: max(0, C.BALANCE - len(v)) for c, v in by_cls.items() if v}
    total = sum(deficit.values())
    if total > C.BALANCE_MAX_AUG:
        deficit = {c: int(d * C.BALANCE_MAX_AUG / total) for c, d in deficit.items()}
    if not sum(deficit.values()):
        return []
    (fdir / "aug").mkdir(parents=True, exist_ok=True)
    rng = stable_rng("balance", seed, fold)
    tf = strong_aug()
    np.random.seed(rng.randint(0, 2**31 - 1))  # albumentations memakai RNG numpy/python
    import random as _r
    _r.seed(rng.randint(0, 2**31 - 1))
    extras, k = [], 0
    for c, d in deficit.items():
        for _ in range(d):
            src = rng.choice(by_cls[c])
            img = load_image(src)
            h, w = img.shape[:2]
            masks, cls = seg_masks(src["segs"], h, w)
            aimg, amasks = apply_aug(tf, img, masks)
            segs = masks_to_segs(amasks, cls)
            if masks and not segs:
                continue
            out = fdir / "aug" / f"{k:05d}.npy"
            np.save(out, _encode(aimg))
            extras.append({"id": f"aug{k}", "src": str(out), "file": f"aug_{k:05d}_{src['id']}",
                           "h": aimg.shape[0], "w": aimg.shape[1], "segs": segs, "dom": c, "aug_of": src["id"]})
            k += 1
    log.info(f"[DATA] {fold_name(seed, fold)} balancing: +{len(extras)} gambar aug "
             f"(deficit {deficit})")
    return extras


def split_summary(fd):
    """Baris ringkasan jumlah gambar & instance per kelas per split (untuk Excel)."""
    rows = []
    for split, its in (("train_orig", fd.train_orig), ("train_aug", fd.extras), ("train_total", fd.train),
                       ("val", fd.val), ("test", fd.test)):
        for c in list(range(fd.nc)) + [fd.nc]:
            rows.append({"seed": fd.seed, "fold": fd.fold, "split": split,
                         "class_id": c, "class": fd.names.get(c, "(tanpa objek)"),
                         "n_images_dominant": sum(1 for it in its if it["dom"] == c),
                         "n_instances": sum(1 for it in its for s in it["segs"] if s[0] == c),
                         "n_images_split": len(its)})
    return rows


# =============================================================================
# LETTERBOX (top-left: skala seragam → ruang S dan ruang EVAL_SIZE proporsional persis)
# =============================================================================
def letterbox(img, S, interp=cv2.INTER_LINEAR, pad=PAD_VALUE):
    h, w = img.shape[:2]
    s = S / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp) if (nh, nw) != (h, w) else img
    out = np.full((S, S) + img.shape[2:], pad, dtype=img.dtype)
    out[:nh, :nw] = r
    return out


# =============================================================================
# DATASET PyTorch
# =============================================================================
class SegDataset:
    """Return: image uint8 (3,S,S) RGB — normalisasi dilakukan DI DALAM model (ONNX-friendly),
    sem (S,S) long (0=background, c+1=kelas), inst_masks (N,S,S) uint8, inst_cls (N,) long."""

    def __init__(self, items, imgsz, train=False, instances=False):
        self.items, self.S, self.train, self.instances = items, imgsz, train, instances
        self.tf = train_aug() if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        it = self.items[i]
        img = load_image(it)
        h, w = img.shape[:2]
        masks, cls = seg_masks(it["segs"], h, w)
        if self.train:
            img, masks = apply_aug(self.tf, img, masks)
        img = letterbox(img, self.S)
        keep = []
        for m, c in zip(masks, cls):
            lm = letterbox(m, self.S, cv2.INTER_NEAREST, 0)
            if lm.any():
                keep.append((lm, c))
        sem = np.zeros((self.S, self.S), np.int64)
        for lm, c in sorted(keep, key=lambda x: -int(x[0].sum())):  # besar dulu, kecil di atas
            sem[lm > 0] = c + 1
        out = {"image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
               "sem": torch.from_numpy(sem), "idx": i}
        if self.instances:
            out["inst_masks"] = torch.from_numpy(np.stack([k[0] for k in keep])) if keep else \
                torch.zeros((0, self.S, self.S), dtype=torch.uint8)
            out["inst_cls"] = torch.tensor([k[1] for k in keep], dtype=torch.long)
        return out


def collate(batch):
    import torch
    out = {"image": torch.stack([b["image"] for b in batch]), "sem": torch.stack([b["sem"] for b in batch]),
           "idx": [b["idx"] for b in batch]}
    if "inst_masks" in batch[0]:
        out["inst_masks"] = [b["inst_masks"] for b in batch]
        out["inst_cls"] = [b["inst_cls"] for b in batch]
    return out


def make_loader(items, imgsz, batch, train, instances=False, workers=None, seed=0):
    import torch
    ds = SegDataset(items, imgsz, train=train, instances=instances)
    g = torch.Generator()
    g.manual_seed(seed)
    w = C.NUM_WORKERS if workers is None else workers
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=w, collate_fn=collate, drop_last=train and len(ds) > batch,
        pin_memory=torch.cuda.is_available(), persistent_workers=False, generator=g,
        worker_init_fn=lambda wid: np.random.seed((seed * 1000 + wid) % 2**31))


# =============================================================================
# GROUND TRUTH evaluasi (ruang letterbox EVAL_SIZE)
# =============================================================================
def eval_gt(items, E=None):
    E = E or C.EVAL_SIZE
    gts = []
    for it in items:
        s = E / max(it["h"], it["w"])
        masks, cls = [], []
        for c, pts in it["segs"]:
            m = np.zeros((E, E), np.uint8)
            p = (np.asarray(pts, np.float32).reshape(-1, 2) * [it["w"] * s, it["h"] * s]).round().astype(np.int32)
            cv2.fillPoly(m, [p], 1)
            if m.any():
                masks.append(m.astype(bool))
                cls.append(int(c))
        gts.append({"masks": np.stack(masks) if masks else np.zeros((0, E, E), bool),
                    "cls": np.array(cls, np.int64)})
    return gts


# =============================================================================
# DATASET YOLO sementara → datasets/s{seed}_f{fold}/
# =============================================================================
def materialize_yolo(fd):
    root = C.DATASETS / fd.name
    yaml_p = root / "data.yaml"
    if yaml_p.exists():
        return yaml_p
    if root.exists():
        shutil.rmtree(root)
    for split, its in (("train", fd.train), ("val", fd.val), ("test", fd.test)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for it in its:
            stem = f"{it['id']}"
            arr = np.load(it["src"])
            ip = root / "images" / split / f"{stem}.jpg"
            if arr.ndim == 1:
                ip.write_bytes(arr.tobytes())   # byte JPEG langsung, tanpa re-encode
            else:
                cv2.imwrite(str(ip), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            (root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(f"{c} " + " ".join(f"{v:.6f}" for v in pts) for c, pts in it["segs"]))
    names = "\n".join(f"  {k}: {v}" for k, v in fd.names.items())
    yaml_p.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return yaml_p


def split_items(fd, split):
    return {"train": fd.train, "val": fd.val, "test": fd.test}[split]
