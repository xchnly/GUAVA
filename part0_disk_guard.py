"""
part0_disk_guard.py — Utility hemat disk WAJIB (dipanggil di awal & akhir tiap part).

Jalankan mandiri untuk laporan disk + bersih-bersih:
    python part0_disk_guard.py            # report
    python part0_disk_guard.py --clean    # clean_all() lalu report

Yang DIHAPUS otomatis:
  runs/*/weights/last.pt, epoch*.pt, results.csv, *.png, train_batch*.jpg, ...
  cache/s*_f*/ fold lama, datasets/s*_f*/ fold lama
  ~/.cache/torch/hub/checkpoints/* & ~/.cache/huggingface/hub/* (kecuali model aktif)
  data_raw/*.jpg setelah cache master jadi
  *.onnx / *.engine / *.xml / *.bin intermediate di luar artifacts/CHAMPION_EDGE
  __pycache__, *.pyc, wandb/, tensorboard/
Yang DISIMPAN: weights/best/**, results/*.xlsx, MASTER_RESULTS.xlsx, state/, logs/, artifacts/CHAMPION_EDGE/
"""
import gc
import gzip
import logging
import os
import shutil
import sys
from pathlib import Path

try:  # konfigurasi global (fallback kalau dipakai di luar project)
    import importlib
    _C = importlib.import_module("00_config")
except Exception:  # pragma: no cover
    _C = None

DISK_MIN_FREE_GB = getattr(_C, "DISK_MIN_FREE_GB", 2.0)
DISK_HARD_STOP_GB = getattr(_C, "DISK_HARD_STOP_GB", 0.5)
DISK_TIER2_PURGE_GB = getattr(_C, "DISK_TIER2_PURGE_GB", 4.0)
ROOT = Path(getattr(_C, "ROOT", "/workspace/guava"))
RUNS_DIR = Path(getattr(_C, "RUNS_DIR", ROOT / "runs"))
CACHE_DIR = Path(getattr(_C, "CACHE_DIR", ROOT / "cache"))
DATASETS = Path(getattr(_C, "DATASETS", ROOT / "datasets"))
LOG_DIR = Path(getattr(_C, "LOG_DIR", ROOT / "logs"))
WEIGHTS_DIR = Path(getattr(_C, "WEIGHTS_DIR", ROOT / "weights"))
ARTIFACTS = Path(getattr(_C, "ARTIFACTS", ROOT / "artifacts"))
TMP_DIR = Path(getattr(_C, "TMP_DIR", ROOT / "tmp"))
RAW_DIR = Path(getattr(_C, "RAW_DIR", ROOT / "data_raw"))
STATE_DIR = Path(getattr(_C, "STATE_DIR", ROOT / "state"))
TIER2_FAMILIES = getattr(_C, "TIER2_FAMILIES", ["transformer", "cnn"])

TORCH_HUB = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

log = logging.getLogger("guava")

RUN_JUNK_PATTERNS = [
    "last.pt", "epoch*.pt", "results.csv", "results.png", "*.png", "train_batch*.jpg",
    "val_batch*.jpg", "labels*.jpg", "*.jpg", "events.out.tfevents*", "args.yaml",
]
EXPORT_PATTERNS = ["*.onnx", "*.engine", "*.trt", "*.tflite", "*_openvino_model", "*.onnx.data"]


class DiskHardStop(RuntimeError):
    """Disk di bawah DISK_HARD_STOP_GB — part harus berhenti & simpan state."""
    exit_code = 3


# -----------------------------------------------------------------------------
# Ukuran disk
# -----------------------------------------------------------------------------
def _probe_path():
    p = ROOT
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def disk_free_gb(path=None):
    return shutil.disk_usage(path or _probe_path()).free / 1e9


def disk_used_gb(path=None):
    return shutil.disk_usage(path or _probe_path()).used / 1e9


def _du(path):
    path = Path(path)
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dp, _, fns in os.walk(path, onerror=lambda e: None):
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def report_disk(top_n=10, root=None):
    """Cetak ringkasan disk + top-N entri terbesar di bawah root."""
    root = Path(root or ROOT)
    total = shutil.disk_usage(_probe_path())
    lines = [f"[DISK] total={total.total/1e9:.1f}GB used={total.used/1e9:.1f}GB free={total.free/1e9:.2f}GB"]
    if root.exists():
        entries = []
        for child in root.iterdir():
            entries.append((_du(child), child))
            if child.is_dir():
                for sub in child.iterdir():
                    entries.append((_du(sub), sub))
        entries.sort(key=lambda x: -x[0])
        for size, p in entries[:top_n]:
            lines.append(f"   {size/1e9:8.3f} GB  {p}")
    for extra in (TORCH_HUB, HF_HUB):
        if extra.exists():
            lines.append(f"   {_du(extra)/1e9:8.3f} GB  {extra}")
    msg = "\n".join(lines)
    if log.handlers:
        log.info(msg)
    else:
        print(msg)
    return total.free / 1e9


# -----------------------------------------------------------------------------
# Helper hapus
# -----------------------------------------------------------------------------
def _rm(p):
    p = Path(p)
    try:
        if p.is_symlink() or p.is_file():
            sz = p.stat().st_size if p.is_file() else 0
            p.unlink(missing_ok=True)
            return sz
        if p.is_dir():
            sz = _du(p)
            shutil.rmtree(p, ignore_errors=True)
            return sz
    except OSError as e:
        log.warning(f"gagal hapus {p}: {e}")
    return 0


def _matches_any(name, tokens):
    name = name.lower()
    return any(t and t.lower() in name for t in tokens)


# -----------------------------------------------------------------------------
# Cleaner spesifik
# -----------------------------------------------------------------------------
def clean_ultralytics_runs(runs_dir=RUNS_DIR):
    """Hapus semua artefak run ultralytics KECUALI weights/best.pt."""
    runs_dir = Path(runs_dir)
    freed = 0
    if not runs_dir.exists():
        return 0
    for p in list(runs_dir.rglob("*")):
        if not p.exists():
            continue
        if p.is_file() and p.name != "best.pt":
            freed += _rm(p)
    for d in sorted([d for d in runs_dir.rglob("*") if d.is_dir()], key=lambda x: -len(str(x))):
        try:
            d.rmdir()  # hanya kalau kosong
        except OSError:
            pass
    return freed


def clean_torch_cache(keep_models=()):
    freed = 0
    if TORCH_HUB.exists():
        for f in TORCH_HUB.iterdir():
            if not _matches_any(f.name, keep_models):
                freed += _rm(f)
    return freed


def _hf_tokens(keep):
    out = []
    for k in keep:
        out += [k, k.replace("/", "--")]
    return out


def clean_hf_cache(keep_repos=()):
    freed = 0
    if HF_HUB.exists():
        toks = _hf_tokens(keep_repos)
        for d in HF_HUB.iterdir():
            if d.name.startswith(("models--", "datasets--")) and not _matches_any(d.name, toks):
                freed += _rm(d)
    return freed


def clean_pretrained(keep_models=()):
    freed = 0
    pdir = WEIGHTS_DIR / "pretrained"
    if pdir.exists():
        for f in pdir.iterdir():
            if not _matches_any(f.name, keep_models):
                freed += _rm(f)
    return freed


def clean_npy_cache(cache_dir=CACHE_DIR, keep_folds=()):
    """Hapus cache/s{seed}_f{fold}/ yang tidak dipakai lagi. cache/master TIDAK dihapus."""
    freed = 0
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        for d in cache_dir.iterdir():
            if d.is_dir() and d.name.startswith("s") and "_f" in d.name and d.name not in keep_folds:
                freed += _rm(d)
    return freed


def clean_datasets(datasets_dir=DATASETS, keep_folds=()):
    freed = 0
    datasets_dir = Path(datasets_dir)
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.name not in keep_folds:
                freed += _rm(d)
    return freed


def clean_pycache(root=None):
    freed = 0
    for base in {Path(root or ROOT), Path(__file__).resolve().parent}:
        if not base.exists():
            continue
        for p in base.rglob("__pycache__"):
            freed += _rm(p)
        for p in base.rglob("*.pyc"):
            freed += _rm(p)
    return freed


def clean_raw_jpg(raw_dir=RAW_DIR, keep=False):
    """Hapus .jpg mentah setelah cache master (.npy) jadi."""
    if keep:
        return 0
    raw_dir = Path(raw_dir)
    master_ok = (CACHE_DIR / "master" / "index.json").exists()
    if not raw_dir.exists() or not master_ok:
        return 0
    freed = 0
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            freed += _rm(p)
    return freed


def clean_intermediate_exports(root=None):
    """Hapus *.onnx/*.engine/openvino intermediate di luar artifacts/CHAMPION_EDGE."""
    freed = 0
    root = Path(root or ROOT)
    champ = (ARTIFACTS / "CHAMPION_EDGE").resolve()
    for pat in EXPORT_PATTERNS:
        for p in root.rglob(pat):
            try:
                if champ in p.resolve().parents or p.resolve() == champ:
                    continue
            except OSError:
                continue
            freed += _rm(p)
    return freed


def clean_loggers_junk(root=None):
    freed = 0
    root = Path(root or ROOT)
    for name in ("wandb", "tensorboard", "mlruns", "lightning_logs"):
        for p in root.rglob(name):
            if p.is_dir():
                freed += _rm(p)
    return freed


def clean_tmp():
    freed = 0
    if TMP_DIR.exists():
        for p in TMP_DIR.iterdir():
            freed += _rm(p)
    return freed


def rotate_logs(log_dir=LOG_DIR, max_mb=None):
    """gzip log > max_mb; hapus .gz tertua bila total log > max_mb."""
    max_mb = max_mb or getattr(_C, "LOG_MAX_MB", 100)
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return 0
    for f in log_dir.glob("*.log"):
        if f.stat().st_size > max_mb * 1e6 / 2:
            gz = f.with_suffix(f".{int(f.stat().st_mtime)}.log.gz")
            with open(f, "rb") as src, gzip.open(gz, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with open(f, "w"):
                pass  # truncate (handler tetap valid)
    gzs = sorted(log_dir.glob("*.gz"), key=lambda p: p.stat().st_mtime)
    freed = 0
    while gzs and _du(log_dir) > max_mb * 1e6:
        freed += _rm(gzs.pop(0))
    return freed


def emergency_tier2_purge():
    """Disk < DISK_TIER2_PURGE_GB → hapus best.pt Tier 2 (Transformer/CNN), Excel tetap."""
    freed = 0
    for fam in TIER2_FAMILIES:
        freed += _rm(WEIGHTS_DIR / "best" / fam)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "tier2_purged.flag").write_text("1")
    log.warning(f"[DISK] EMERGENCY: best.pt Tier 2 dihapus ({freed/1e9:.2f} GB)")
    return freed


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def clean_all(runs_dir=RUNS_DIR, cache_dir=CACHE_DIR, keep_models=(), keep_folds=(),
              purge_model_cache=True, verbose=True):
    """Panggil semua cleaner + gc.collect + torch.cuda.empty_cache.

    keep_models: token nama (mis. "yolo11x-seg", "nvidia/segformer-b5") yang cache-nya dipertahankan.
    keep_folds : nama folder fold (mis. "s42_f0") yang cache/dataset-nya dipertahankan.
    purge_model_cache=False → cache HF/torch/pretrained tidak disentuh.
    """
    before = disk_free_gb()
    clean_ultralytics_runs(runs_dir)
    if purge_model_cache:
        clean_torch_cache(keep_models)
        clean_hf_cache(keep_models)
        clean_pretrained(keep_models)
    clean_npy_cache(cache_dir, keep_folds)
    clean_datasets(DATASETS, keep_folds)
    clean_pycache()
    clean_raw_jpg(RAW_DIR)
    clean_intermediate_exports()
    clean_loggers_junk()
    clean_tmp()
    rotate_logs()
    _free_gpu()
    after = disk_free_gb()
    if verbose:
        log.info(f"[DISK] clean_all: free {before:.2f} → {after:.2f} GB "
                 f"(keep_models={list(keep_models)[:6]}, keep_folds={list(keep_folds)})")
    return after - before


def vram_status():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return None, None


def guard_before_training(keep_models=(), keep_folds=(), min_vram_gb=1.0):
    """Cek disk + VRAM sebelum training.

    - free < DISK_TIER2_PURGE_GB → emergency_tier2_purge()
    - free < DISK_MIN_FREE_GB    → clean_all() dulu
    - free < DISK_HARD_STOP_GB   → raise DiskHardStop (RuntimeError)
    """
    free = disk_free_gb()
    if free < DISK_MIN_FREE_GB:
        log.warning(f"[DISK] free {free:.2f} GB < {DISK_MIN_FREE_GB} → clean_all()")
        clean_all(keep_models=keep_models, keep_folds=keep_folds)
        free = disk_free_gb()
    if free < DISK_TIER2_PURGE_GB and not (STATE_DIR / "tier2_purged.flag").exists():
        emergency_tier2_purge()
        free = disk_free_gb()
    if free < DISK_HARD_STOP_GB:
        raise DiskHardStop(f"Disk free {free:.2f} GB < HARD STOP {DISK_HARD_STOP_GB} GB")
    vf, vt = vram_status()
    if vf is not None:
        if vf < min_vram_gb:
            _free_gpu()
            vf, vt = vram_status()
        log.info(f"[GUARD] disk_free={free:.2f}GB vram_free={vf:.2f}/{vt:.2f}GB")
        if vf < min_vram_gb:
            log.warning(f"[GUARD] VRAM free hanya {vf:.2f} GB — ada proses lain di GPU?")
    else:
        log.info(f"[GUARD] disk_free={free:.2f}GB (tanpa GPU)")
    return free


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if "--clean" in sys.argv:
        clean_all(purge_model_cache="--keep-cache" not in sys.argv)
    report_disk(top_n=15)
