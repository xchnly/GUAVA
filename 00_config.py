"""
00_config.py — Konfigurasi global eksperimen segmentasi guava.

TIDAK ADA kode berat di sini. Semua part meng-import file ini via:

    from guava_core import C      # C = modul 00_config

Nama file diawali angka, jadi tidak bisa `import 00_config` secara langsung;
guava_core memakai importlib.import_module("00_config").

Override cepat lewat environment variable (berguna di Vast.ai):
    GUAVA_ROOT=/workspace/guava
    GUAVA_PROFILE=full | fast | smoke
    GUAVA_SEEDS=42,123        (subset seed)
    GUAVA_FOLDS=0,1           (subset fold)
"""
import os
from pathlib import Path


def _env_list(name, default, cast=int):
    v = os.environ.get(name, "").strip()
    return [cast(x) for x in v.split(",") if x.strip()] if v else list(default)


# =============================================================================
# PATH
# =============================================================================
ROOT = Path(os.environ.get("GUAVA_ROOT", "/workspace/guava"))
NDJSON_PATH = Path(os.environ.get("GUAVA_NDJSON", ROOT / "guava.ndjson"))
DATASETS = ROOT / "datasets"          # dataset YOLO sementara per (seed, fold)
RUNS_DIR = ROOT / "runs"              # run ultralytics sementara
ARTIFACTS = ROOT / "artifacts"        # CHAMPION_EDGE/
CACHE_DIR = ROOT / "cache"            # master/ + s{seed}_f{fold}/ (.npy uint8)
EXCEL_DIR = ROOT / "results"          # part*.xlsx + summary_plots/
STATE_DIR = ROOT / "state"            # resume state + journal
LOG_DIR = ROOT / "logs"
RAW_DIR = ROOT / "data_raw"           # .jpg mentah (dihapus setelah cache jadi)
WEIGHTS_DIR = ROOT / "weights"        # pretrained/ work/ best/
TMP_DIR = ROOT / "tmp"                # cwd ultralytics, file sementara
MASTER_XLSX = ROOT / "MASTER_RESULTS.xlsx"

# =============================================================================
# PROFILE (full = sesuai spesifikasi riset)
# =============================================================================
PROFILE = os.environ.get("GUAVA_PROFILE", "full").lower()

# =============================================================================
# EXPERIMENT
# =============================================================================
SEEDS = _env_list("GUAVA_SEEDS", [42, 123, 2024, 7, 999])
N_FOLDS = 5
FOLDS = _env_list("GUAVA_FOLDS", list(range(N_FOLDS)))
SPLIT = (0.80, 0.10, 0.10)
# Skema: per seed, data dipecah stratified jadi 10 chunk (StratifiedKFold n=10).
# Fold k: test = chunk 2k, val = chunk 2k+1, train = 8 chunk sisanya  → 80/10/10.
# Setiap gambar tepat sekali jadi val ATAU test dalam satu seed.
IMGSZ_CHOICES = _env_list("GUAVA_IMGSZ", [512, 640, 768])  # multi-resolusi, random per run (deterministik)
MASTER_MAX_SIDE = max(IMGSZ_CHOICES)  # resolusi cache master (long side)
BALANCE = 400                         # target gambar per kelas di train
BALANCE_MAX_AUG = 3000                # maksimal tambahan gambar augmentasi
CACHE_FORMAT = "jpgbytes"             # "jpgbytes" (npy uint8 berisi byte JPEG, hemat disk)
                                      # atau "raw" (npy uint8 HxWx3, boros disk ~10x)
CACHE_JPEG_QUALITY = 95
EVAL_SIZE = 640                       # semua model dievaluasi di ruang letterbox 640
EVAL_MAX_DET = 100
EVAL_MIN_AREA = 16                    # komponen < 16 px diabaikan (model semantik)
NUM_WORKERS = min(8, os.cpu_count() or 4)

# =============================================================================
# BATCH per family (konservatif untuk 16 GB VRAM)
# =============================================================================
BATCH_YOLO = 8
BATCH_TRANSFORMER = 2
BATCH_CNN = 16
BATCH_EDGE = 32

# =============================================================================
# EPOCHS per family
# =============================================================================
EPOCHS_YOLO = 150
EPOCHS_TRANSFORMER = 80
EPOCHS_CNN = 100
EPOCHS_EDGE = 120
PATIENCE = 30
VAL_EVERY = 1                         # evaluasi val tiap N epoch (non-YOLO)

LR = {"transformer": 6e-5, "cnn": 5e-4, "edge": 1e-3}
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
AMP_DTYPE = "bf16"                    # RTX 50xx (Blackwell) mendukung bf16
GRAD_CKPT_TRANSFORMER = True
MAX_RETRY = 2                         # unit yang crash dicoba ulang maks 2x, lalu di-skip

# =============================================================================
# MODEL SET
# =============================================================================
# Hanya model yang TERVERIFIKASI bisa di-build + forward/backward (lihat guava_models.preflight).
# Dibuang: Mask2Former (jalur loss/eval khusus, rapuh), DPT-Large (340M param, mepet VRAM & disk),
#          MobileSAM (encoder terkunci input 1024), FastSAM (arsitektur = YOLOv8-seg).
# Saat runtime, model yang gagal preflight (mis. bobot gagal diunduh) otomatis di-skip,
# tapi minimal 1 model per family (yolo / transformer / cnn / edge) WAJIB lolos.
YOLO_MODELS = _env_list("GUAVA_YOLO_MODELS", ["yolo11x-seg", "yolo26x-seg"], str)
TRANSFORMER_MODELS = _env_list("GUAVA_TRANSFORMER_MODELS", ["segformer-b5", "segformer-b2"], str)
CNN_MODELS = _env_list("GUAVA_CNN_MODELS", ["mnv3L-deeplabv3plus", "mnv2-unet", "effb0-unet"], str)
EDGE_MODELS = _env_list("GUAVA_EDGE_MODELS", [  # 4 tambahan + model ringan Part 2 (profil edge)
    "mnv3S-lraspp", "efflite0-unet", "shufflenetv2-unet", "mobileone-s0-unet",
    "mnv2-unet", "mnv3L-deeplabv3plus",
], str)

# =============================================================================
# KD (Part 4)
# =============================================================================
KD_STUDENTS = _env_list("GUAVA_KD_STUDENTS",
                        ["mnv3S-lraspp", "efflite0-unet", "shufflenetv2-unet", "mobileone-s0-unet", "mnv2-unet"], str)
KD_ALPHAS = [0.3, 0.5, 0.7]
KD_TEMPS = [2, 4, 6]
KD_BETA = 0.3          # weight logits KD
KD_GAMMA = 0.1         # weight feature KD
KD_BETA_FROM_ALPHA = False   # True → beta = 1 - alpha (formulasi Hinton klasik)
# "full"             : semua (alpha,T) di semua 5 seed × 5 fold (sesuai spesifikasi, 1125 run)
# "search_then_full" : grid penuh hanya di SEEDS[0]; seed lain pakai (alpha,T) terbaik per student
KD_GRID_SCOPE = "full"
# Teacher dipilih per (seed, fold) dari model Part 1/2/3 yang dilatih di split yang SAMA
# (mencegah leakage). Metrik seleksi: val (bukan test) agar test tetap "unseen".
TEACHER_SELECT_METRIC = "val_mAP50-95_mask"
TEACHER_PARTS = ["part1", "part2", "part3"]
KD_TEACHER_CHUNK = 8     # sub-batch forward teacher (hemat VRAM)
KD_TEACHER_CONF = 0.10

# =============================================================================
# PRUNING / QUANTIZATION (Part 5)
# =============================================================================
PRUNE_RATIOS = [0.2, 0.4, 0.6]
PRUNE_FT_EPOCHS = 15
PRUNE_FT_LR = 1e-4
P5_MODELS = list(KD_STUDENTS)
P5_SOURCES = ["baseline", "kd"]     # prune/quant bobot baseline (Part 3) & KD terbaik (Part 4)
QAT_N_MODELS = 2
QAT_EPOCHS = 5
QAT_LR = 5e-5
CALIB_IMAGES = 64

# =============================================================================
# CHAMPION (Part 6)
# =============================================================================
CHAMP_W = (0.6, 0.2, 0.2)          # w1 mAP, w2 latency, w3 size
ONNX_OPSET = 12
EXPORT_TFLITE = False              # butuh onnx2tf + tensorflow (berat)
EXPORT_TENSORRT = True

# =============================================================================
# LATENCY
# =============================================================================
LAT_WARMUP = 3
LAT_RUNS = 20
LAT_RUNS_BIG = 5                   # model > 50M param (CPU sangat lambat)
LAT_CPU_THREADS = 4

# =============================================================================
# DISK
# =============================================================================
DISK_MIN_FREE_GB = 2.0
DISK_HARD_STOP_GB = 0.5
DISK_TIER2_PURGE_GB = 4.0          # < 4 GB → hapus best.pt Tier 2 (transformer/cnn), tunda Part 5
CACHE_PURGE_FREE_GB = 6.0          # cache HF/torch dihapus total di akhir part hanya jika free < 6 GB
LOG_MAX_MB = 100
CLEAN_EVERY_N_RUNS = 1
# Retensi bobot final (weights/best/...):
#   per_fold  → 1 file per (model, fold), terbaik lintas seed (by val)
#   per_model → 1 file per model, terbaik lintas seed & fold
RETENTION = {
    "yolo": "per_fold", "transformer": "per_model", "cnn": "per_fold",
    "edge": "per_fold", "kd": "per_model", "prune": "per_model",
}
TIER2_FAMILIES = ["transformer", "cnn"]

# =============================================================================
# ORCHESTRATOR (Part 8)
# =============================================================================
# Loop (seed, fold) di luar; part di dalam — hanya 1 set bobot kerja di disk.
ORCH_ORDER = ["part1", "part3", "part4", "part2", "part5"]   # P1 → P2
ORCH_FINAL = ["part6", "part7"]                              # P3
PART_SCRIPTS = {
    "part1": "part1_dataset_baseline.py",
    "part2": "part2_transformer_cnn.py",
    "part3": "part3_edge_candidates.py",
    "part4": "part4_compression_kd.py",
    "part5": "part5_compression_prune_quant.py",
    "part6": "part6_champion_export.py",
    "part7": "part7_excel_reporter.py",
}

# =============================================================================
# PROFILE override
# =============================================================================
if PROFILE == "fast":
    EPOCHS_YOLO, EPOCHS_TRANSFORMER, EPOCHS_CNN, EPOCHS_EDGE = 60, 30, 40, 50
    PATIENCE = 15
    PRUNE_FT_EPOCHS, QAT_EPOCHS = 8, 3
    KD_GRID_SCOPE = "search_then_full"
elif PROFILE == "smoke":            # uji pipeline end-to-end dalam hitungan menit
    SEEDS = SEEDS[:1]
    FOLDS = FOLDS[:1]
    EPOCHS_YOLO = EPOCHS_TRANSFORMER = EPOCHS_CNN = EPOCHS_EDGE = 1
    PATIENCE = 1
    PRUNE_FT_EPOCHS = QAT_EPOCHS = 1
    KD_ALPHAS, KD_TEMPS = [0.5], [4]
    PRUNE_RATIOS = [0.4]
    BALANCE, BALANCE_MAX_AUG = 20, 20
    LAT_RUNS, LAT_RUNS_BIG = 2, 1
    CALIB_IMAGES = 4
