"""
guava_core.py — Infrastruktur bersama semua part:
config, logging, seed, state resume, Excel incremental (journal), manajemen bobot,
pengukuran params/FLOPs/size/latency, dan runner unit dengan retry.
"""
import argparse
import copy
import importlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from datetime import datetime
from pathlib import Path

C = importlib.import_module("00_config")
import part0_disk_guard as DG  # noqa: E402

log = logging.getLogger("guava")

for _d in (C.ROOT, C.DATASETS, C.RUNS_DIR, C.ARTIFACTS, C.CACHE_DIR, C.EXCEL_DIR, C.STATE_DIR,
           C.LOG_DIR, C.RAW_DIR, C.WEIGHTS_DIR, C.TMP_DIR):
    Path(_d).mkdir(parents=True, exist_ok=True)

# Matikan logger eksternal yang menulis artefak ke disk
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("COMET_MODE", "disabled")
os.environ.setdefault("CLEARML_OFF", "1")
os.environ.setdefault("PIP_NO_CACHE_DIR", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("YOLO_VERBOSE", "False")

EXIT_DISK = 3

# =============================================================================
# Dependencies
# =============================================================================
DEPS = {  # import name → pip name
    "ultralytics": "ultralytics", "albumentations": "albumentations",
    "segmentation_models_pytorch": "segmentation-models-pytorch", "transformers": "transformers",
    "timm": "timm", "onnx": "onnx", "onnxruntime": "onnxruntime", "openvino": "openvino",
    "nncf": "nncf", "pandas": "pandas", "openpyxl": "openpyxl", "xlsxwriter": "xlsxwriter",
    "sklearn": "scikit-learn", "psutil": "psutil", "GPUtil": "GPUtil", "thop": "thop",
    "pyarrow": "pyarrow", "torch_pruning": "torch-pruning", "scipy": "scipy",
    "matplotlib": "matplotlib", "cv2": "opencv-python-headless", "onnxsim": "onnxsim",
}


def ensure_deps(names=None):
    """Install paket yang belum ada (pip --no-cache-dir → hemat disk). torch TIDAK di-install
    otomatis: RTX 50xx butuh build cu128+, gunakan image Vast.ai PyTorch terbaru."""
    names = names or list(DEPS)
    missing = []
    for mod in names:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(DEPS.get(mod, mod))
    if missing:
        log.info(f"[DEPS] install: {missing}")
        for pkg in missing:  # satu-satu: satu paket gagal tidak menggagalkan yang lain
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", pkg],
                               capture_output=True, text=True)
            if r.returncode != 0:
                log.warning(f"[DEPS] gagal install {pkg}: {r.stderr[-300:]}")
    return missing


# =============================================================================
# Logging, seed, CLI
# =============================================================================
def setup_logging(part):
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | " + part + " | %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.LOG_DIR / f"{part}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    for noisy in ("PIL", "matplotlib", "urllib3", "httpx", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True   # multi-resolusi tetap diuntungkan
    except Exception:
        pass


def stable_rng(*parts):
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def pick_imgsz(seed, fold, model):
    """Resolusi random per run, deterministik per (seed, fold, model) → baseline, KD, prune
    dari student yang sama memakai imgsz yang sama (perbandingan adil)."""
    return stable_rng("imgsz", seed, fold, model).choice(C.IMGSZ_CHOICES)


def fold_name(seed, fold):
    return f"s{seed}_f{fold}"


def parse_part_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--seed", type=int, action="append", help="batasi ke seed ini (boleh berulang)")
    ap.add_argument("--fold", type=int, action="append", help="batasi ke fold ini (boleh berulang)")
    ap.add_argument("--no-final", action="store_true",
                    help="dipakai orchestrator: jangan tulis partN_done.json / clean total")
    ap.add_argument("--models", type=str, default="", help="subset model, pisah koma")
    args = ap.parse_args()
    args.seeds = args.seed or list(C.SEEDS)
    args.folds = args.fold or list(C.FOLDS)
    args.model_list = [m for m in args.models.split(",") if m]
    return args


def env_report():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["capability"] = torch.cuda.get_device_capability(0)
    except Exception as e:
        info["torch"] = f"ERR {e}"
    info["disk_free_GB"] = round(DG.disk_free_gb(), 2)
    log.info(f"[ENV] {info}")
    return info


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# JSON atomik
# =============================================================================
def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return copy.deepcopy(default)


# =============================================================================
# STATE (resume)
# =============================================================================
class PartState:
    """state/{part}.json — unit selesai, jumlah percobaan, error terakhir, data tambahan."""

    def __init__(self, part):
        self.part = part
        self.path = C.STATE_DIR / f"{part}.json"
        self.data = read_json(self.path, {"done": {}, "attempts": {}, "errors": {}, "extra": {}})
        for k in ("done", "attempts", "errors", "extra"):
            self.data.setdefault(k, {})

    def save(self):
        write_json(self.path, self.data)

    def is_done(self, key):
        return key in self.data["done"]

    def attempts(self, key):
        return self.data["attempts"].get(key, 0)

    def start(self, key):
        self.data["attempts"][key] = self.attempts(key) + 1
        self.save()
        return self.data["attempts"][key]

    def mark_done(self, key, status="OK"):
        self.data["done"][key] = {"status": status, "t": now()}
        self.save()

    def mark_error(self, key, err):
        self.data["errors"][key] = str(err)[-2000:]
        self.save()

    def get(self, k, default=None):
        return self.data["extra"].get(k, default)

    def set(self, k, v):
        self.data["extra"][k] = v
        self.save()


def part_done_path(part):
    return C.STATE_DIR / f"{part}_done.json"


def is_part_done(part):
    return part_done_path(part).exists()


def write_part_done(part, state, expected_keys, extra=None):
    """Tulis state/partN_done.json hanya jika semua unit yang diharapkan sudah selesai."""
    missing = [k for k in expected_keys if not state.is_done(k)]
    if missing:
        log.info(f"[STATE] {part}: {len(missing)} unit belum selesai → belum done")
        return False
    n_fail = sum(1 for k in expected_keys if state.data["done"][k]["status"] != "OK")
    write_json(part_done_path(part), {"part": part, "units": len(expected_keys), "failed": n_fail,
                                      "finished": now(), **(extra or {})})
    log.info(f"[STATE] {part} DONE ({len(expected_keys)} unit, {n_fail} gagal)")
    return True


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# EXCEL incremental (journal jsonl → xlsx atomik)
# =============================================================================
class ResultBook:
    """Setiap baris di-append ke journal state/journal/*.jsonl (O(1), crash-safe), lalu
    xlsx ditulis ulang secara atomik: sheet 'results' tiap baris, sheet 'history' tiap
    `history_every` run. Baris duplikat (unit yang di-retry) → ambil attempt terakhir."""

    def __init__(self, xlsx_path, history_every=10):
        self.xlsx = Path(xlsx_path)
        self.jdir = C.STATE_DIR / "journal"
        self.jdir.mkdir(parents=True, exist_ok=True)
        self.history_every = history_every
        self._since_flush = 0

    def _jpath(self, sheet):
        return self.jdir / f"{self.xlsx.stem}__{sheet}.jsonl"

    def add(self, sheet, rows, flush=True):
        rows = rows if isinstance(rows, list) else [rows]
        with open(self._jpath(sheet), "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=_json_default) + "\n")
        if flush:
            self.flush()

    def rows(self, sheet):
        p = self._jpath(sheet)
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # baris terpotong karena crash
        return out

    def frame(self, sheet):
        import pandas as pd
        df = pd.DataFrame(self.rows(sheet))
        if df.empty:
            return df
        if "unit" in df.columns and "attempt" in df.columns:
            df["unit"] = df["unit"].fillna("-")
            df["attempt"] = df["attempt"].fillna(0)
            last = df.groupby("unit")["attempt"].transform("max")
            df = df[df["attempt"] == last]
        if sheet != "history" and "unit" in df.columns:
            df = df.drop_duplicates(subset="unit", keep="last")
        return df.reset_index(drop=True)

    def sheets(self):
        pre = f"{self.xlsx.stem}__"
        return sorted(p.name[len(pre):-6] for p in self.jdir.glob(f"{pre}*.jsonl"))

    @property
    def history_xlsx(self):
        return self.xlsx.parent / "history" / f"{self.xlsx.stem}_history.xlsx"

    def run_finished(self):
        self._since_flush += 1
        if self._since_flush >= self.history_every:
            self.flush(include_history=True)

    def flush(self, include_history=False):
        """Sheet ringkas → results/{part}.xlsx (tiap baris). Sheet 'history' (per-epoch, besar)
        → results/history/{part}_history.xlsx (tiap `history_every` run & akhir part)."""
        sheets = [s for s in self.sheets() if s != "history"]
        targets = [(self.xlsx, sheets)] if sheets else []
        if include_history and "history" in self.sheets():
            self.history_xlsx.parent.mkdir(parents=True, exist_ok=True)
            targets.append((self.history_xlsx, ["history"]))
            self._since_flush = 0
        for path, shs in targets:
            self._write(path, shs)

    def _write(self, path, sheets):
        import pandas as pd
        tmp = path.with_suffix(".tmp.xlsx")
        try:
            with pd.ExcelWriter(tmp, engine="xlsxwriter") as xw:
                for sh in sheets:
                    self.frame(sh).to_excel(xw, sheet_name=sh[:31], index=False)
            os.replace(tmp, path)
        except Exception as e:
            log.warning(f"[EXCEL] gagal tulis {path.name}: {e}")
            tmp.unlink(missing_ok=True)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


def load_results(part_xlsx_stem, sheet="results"):
    """Baca hasil part lain via journal (lebih cepat & selalu terbaru dibanding xlsx)."""
    return ResultBook(C.EXCEL_DIR / f"{part_xlsx_stem}.xlsx").frame(sheet)


# =============================================================================
# BOBOT: work (per seed-fold, sementara) & best (final, retensi)
# =============================================================================
def work_dir(seed, fold):
    d = C.WEIGHTS_DIR / "work" / fold_name(seed, fold)
    d.mkdir(parents=True, exist_ok=True)
    return d


def finalize_fold(seed, fold):
    """Hapus bobot kerja (s,f) setelah semua konsumennya (Part 4/5) selesai."""
    d = C.WEIGHTS_DIR / "work" / fold_name(seed, fold)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        log.info(f"[WEIGHTS] work {fold_name(seed, fold)} dihapus")


def promote_final(src, family, model_key, seed, fold, score, meta=None):
    """Salin bobot ke weights/best/{family}/{model_key}/ bila lebih baik (by val) dari yang ada.
    Retensi sesuai C.RETENTION[family]: per_fold (fold{f}.pt) atau per_model (best.pt)."""
    src = Path(src)
    if not src.exists():
        return None
    if family in C.TIER2_FAMILIES and (C.STATE_DIR / "tier2_purged.flag").exists():
        return None  # disk kritis: Tier 2 hanya disimpan Excel-nya
    mode = C.RETENTION.get(family, "per_fold")
    fname = f"fold{fold}.pt" if mode == "per_fold" else "best.pt"
    dst_dir = C.WEIGHTS_DIR / "best" / family / model_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    idx_path = dst_dir / "index.json"
    idx = read_json(idx_path, {})
    cur = idx.get(fname)
    score = float(score if score == score else -1)  # NaN → -1
    if cur is None or score > cur["score"] or not (dst_dir / fname).exists():
        tmp = dst_dir / (fname + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst_dir / fname)
        idx[fname] = {"score": score, "seed": seed, "fold": fold, "t": now(), **(meta or {})}
        write_json(idx_path, idx)
        log.info(f"[WEIGHTS] best {family}/{model_key}/{fname} ← s{seed} f{fold} score={score:.4f}")
        return dst_dir / fname
    return None


def find_final(family, model_key, prefer_fold=None):
    d = C.WEIGHTS_DIR / "best" / family / model_key
    idx = read_json(d / "index.json", {})
    if not idx:
        return None, None
    if prefer_fold is not None and f"fold{prefer_fold}.pt" in idx:
        fn = f"fold{prefer_fold}.pt"
    else:
        fn = max(idx, key=lambda k: idx[k]["score"])
    p = d / fn
    return (p, idx[fn]) if p.exists() else (None, None)


def update_teacher(seed, fold, model, kind, family, score, src, imgsz, part):
    """Kandidat teacher per (seed, fold): simpan hanya yang terbaik (by val) → 1 file per fold."""
    wd = work_dir(seed, fold)
    meta_p = wd / "teacher.json"
    meta = read_json(meta_p, None)
    score = float(score if score == score else -1)
    if meta is None or score > meta["score"] or not (wd / "teacher.pt").exists():
        tmp = wd / "teacher.pt.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, wd / "teacher.pt")
        write_json(meta_p, {"model": model, "kind": kind, "family": family, "score": score,
                            "imgsz": imgsz, "part": part, "seed": seed, "fold": fold})
        log.info(f"[TEACHER] {fold_name(seed, fold)} → {model} ({C.TEACHER_SELECT_METRIC}={score:.4f})")


def file_mb(p):
    p = Path(p)
    if p.is_dir():
        return DG._du(p) / 1e6
    return p.stat().st_size / 1e6 if p.exists() else float("nan")


# =============================================================================
# PROFILING: params, FLOPs, latency, VRAM
# =============================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, imgsz, dev="cpu"):
    """GFLOPs (= 2 × MACs, konvensi Ultralytics) via thop pada salinan model."""
    import torch
    try:
        import thop
        m = copy.deepcopy(model).to(dev).eval()
        x = torch.zeros(1, 3, imgsz, imgsz, device=dev)
        with torch.no_grad():
            macs, _ = thop.profile(m, inputs=(x,), verbose=False)
        del m
        return 2 * macs / 1e9
    except Exception as e:
        log.warning(f"[FLOPs] thop gagal: {repr(e)[:150]}")
        return float("nan")


def measure_latency(model, imgsz, dev, runs=None, warmup=None, fp16=False):
    """Latency batch=1 (ms, median). Input float 0..255 (normalisasi di dalam model)."""
    import torch
    runs = runs or (C.LAT_RUNS if count_params(model) < 50 else C.LAT_RUNS_BIG)
    warmup = warmup or C.LAT_WARMUP
    dev = torch.device(dev)
    if dev.type == "cpu":
        torch.set_num_threads(C.LAT_CPU_THREADS)
    try:
        m = copy.deepcopy(model).to(dev).eval()
        x = torch.rand(1, 3, imgsz, imgsz, device=dev) * 255
        ctx = torch.autocast("cuda", dtype=torch.float16) if (fp16 and dev.type == "cuda") else _null()
        times = []
        with torch.no_grad(), ctx:
            for i in range(warmup + runs):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                m(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= warmup:
                    times.append((time.perf_counter() - t) * 1000)
        del m
        times.sort()
        return times[len(times) // 2]
    except Exception as e:
        log.warning(f"[LAT] {dev} gagal: {repr(e)[:150]}")
        return float("nan")
    finally:
        torch.set_num_threads(os.cpu_count() or 4)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def vram_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def vram_peak_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return float("nan")


def free_gpu():
    DG._free_gpu()


# =============================================================================
# RUNNER unit (resume + retry + baris Excel segera)
# =============================================================================
def unit_key(model, seed, fold, tech="baseline", hp="-"):
    return f"{model}|s{seed}|f{fold}|{tech}|{hp}"


def run_unit(state, book, key, base_row, fn, keep_models=(), keep_folds=()):
    """Jalankan fn() → dict metrik. Tulis baris ke Excel SEGERA. Retry sampai C.MAX_RETRY.
    Crash keras (proses mati) tetap terhitung karena attempts di-increment sebelum mulai."""
    if state.is_done(key):
        return None
    while True:
        if state.attempts(key) > C.MAX_RETRY:
            row = {**base_row, "unit": key, "attempt": state.attempts(key), "status": "GAVE_UP: "
                   + state.data["errors"].get(key, "crash tanpa pesan (kemungkinan OOM-kill)")[-300:],
                   "timestamp": now()}
            book.add("results", row)
            state.mark_done(key, "FAILED")
            log.error(f"[UNIT] {key} menyerah setelah {C.MAX_RETRY + 1} percobaan")
            return row
        DG.guard_before_training(keep_models=keep_models, keep_folds=keep_folds)
        attempt = state.start(key)
        log.info(f"[UNIT] ▶ {key} (attempt {attempt})")
        t0 = time.time()
        vram_reset()
        try:
            res = fn(attempt) or {}
            row = {**base_row, **res, "unit": key, "attempt": attempt,
                   "status": res.get("status", "OK"),
                   "duration_min": round((time.time() - t0) / 60, 2),
                   "vram_peak_GB": round(vram_peak_gb(), 2), "timestamp": now()}
            book.add("results", row)
            book.run_finished()
            state.mark_done(key, "OK")
            log.info(f"[UNIT] ✔ {key} {row.get('duration_min')} min | "
                     f"val={row.get('val_mAP50-95_mask', float('nan')):.4f} "
                     f"test={row.get('test_mAP50-95_mask', float('nan')):.4f}")
            return row
        except DG.DiskHardStop:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"[UNIT] ✘ {key} attempt {attempt}: {repr(e)}\n{tb[-3000:]}")
            state.mark_error(key, f"{type(e).__name__}: {e}")
            free_gpu()
            DG.clean_ultralytics_runs(C.RUNS_DIR)


def part_main(part, body):
    """Wrapper main(): DiskHardStop → exit 3 (orchestrator berhenti), error lain → exit 1."""
    setup_logging(part)
    try:
        body()
    except DG.DiskHardStop as e:
        log.critical(f"[DISK] HARD STOP: {e}. State tersimpan; bebaskan disk lalu jalankan ulang.")
        sys.exit(EXIT_DISK)
    except KeyboardInterrupt:
        log.warning("dihentikan user")
        sys.exit(130)
    except Exception:
        log.critical(traceback.format_exc())
        sys.exit(1)
