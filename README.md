# GUAVA — Eksperimen Segmentasi Buah Jambu (Instance Seg, YOLO-seg polygon)

Pipeline multi-file, resume-able dan hemat disk untuk Vast.ai (RTX 5060 Ti 16 GB, disk 16 GB):
split ulang 80/10/10 × 5-fold × 5 seed → YOLO / Transformer / CNN / Edge → KD, Pruning,
Quantization → champion edge (non-YOLO) → `MASTER_RESULTS.xlsx`.

## Cara jalan

```bash
# 1. Taruh dataset: /workspace/guava/guava.ndjson  (format NDJSON Ultralytics; URL gambar diunduh otomatis)
#    atau dataset YOLO-seg biasa:  export GUAVA_YOLO_DIR=/path/ke/dataset
# 2. Uji pipeline cepat (1 seed, 1 fold, 1 epoch — beberapa menit):
GUAVA_PROFILE=smoke bash run_all.sh
rm -rf /workspace/guava/{state,results,weights,artifacts,MASTER_RESULTS.xlsx}   # bersihkan hasil smoke
# 3. Jalankan penuh (berhari-hari, resume otomatis):
nohup bash run_all.sh > run.out 2>&1 &
```

Kalau crash / instance restart: **jalankan ulang perintah yang sama** — tiap unit
(model × seed × fold × teknik × hyperparam) yang sudah selesai tercatat di `state/*.json` dan di-skip.
Tiap part juga bisa dijalankan sendiri: `python part4_compression_kd.py --seed 42 --fold 0`.

| File | Isi |
|---|---|
| `00_config.py` | semua konfigurasi (path, seed, epoch, batch, KD/prune grid, disk, retensi bobot) |
| `part0_disk_guard.py` | `disk_free_gb`, `report_disk`, `clean_*`, `clean_all`, `guard_before_training` |
| `part1_dataset_baseline.py` | cache data, split, balancing, `dataset_split_summary.xlsx`, YOLO11x/YOLO26x-seg |
| `part2_transformer_cnn.py` | SegFormer-B5/B2, MobileNetV3-L DeepLabV3+, MobileNetV2-UNet, EfficientNet-B0-UNet |
| `part3_edge_candidates.py` | 6 kandidat edge: params, FLOPs, size, latency CPU/GPU, mAP |
| `part4_compression_kd.py` | KD (fokus riset): grid α × T per student, delta vs baseline |
| `part5_compression_prune_quant.py` | structured pruning + fine-tune, PTQ INT8 (ORT & OpenVINO/NNCF), QAT |
| `part6_champion_export.py` | skor champion → ONNX / INT8 / OpenVINO / TensorRT di `artifacts/CHAMPION_EDGE/` |
| `part7_excel_reporter.py` | `MASTER_RESULTS.xlsx` (+ `kd_significance`, `all_runs_flat`) + plot |
| `part8_orchestrator.py` | otak utama (dipanggil `run_all.sh`) |
| `guava_core.py` / `guava_data.py` / `guava_models.py` / `guava_train.py` | library bersama |

## Model yang dipakai (hanya yang lolos uji)

Semua model di bawah sudah diuji build + forward/backward. Saat runtime, **preflight** mencoba
lagi dengan bobot pretrained; model yang gagal (mis. download gagal) otomatis di-skip dan dicatat di
`state/model_preflight.json`, tapi part berhenti kalau satu family (YOLO/Transformer/CNN/Edge) tidak
punya model tersisa.

| Family | Model |
|---|---|
| YOLO | `yolo11x-seg`, `yolo26x-seg` |
| Transformer | `segformer-b5`, `segformer-b2` |
| CNN | `mnv3L-deeplabv3plus`, `mnv2-unet`, `effb0-unet` |
| Edge / student | `mnv3S-lraspp`, `efflite0-unet`, `shufflenetv2-unet`, `mobileone-s0-unet`, `mnv2-unet`, `mnv3L-deeplabv3plus` |

Dibuang karena bermasalah: Mask2Former (butuh jalur loss/eval khusus yang rapuh), DPT-Large
(340M param, mepet VRAM & disk), MobileSAM (encoder terkunci input 1024), FastSAM (arsitekturnya YOLOv8-seg).
Di Part 5, kombinasi yang tidak didukung di-skip otomatis: ShuffleNetV2 tidak bisa di-prune
(channel shuffle), EfficientNet-Lite0 & DeepLabV3+ tidak bisa di-trace FX untuk QAT.

## Keputusan desain penting

* **Split 80/10/10 × 5-fold**: per seed, `StratifiedKFold(10)` (stratified per kelas dominan) → fold k:
  test = chunk 2k, val = chunk 2k+1, train = 8 chunk sisanya. Tiap gambar tepat sekali jadi val/test per seed.
* **Evaluator terpadu**: semua family dinilai dengan kode COCO-101 yang sama di ruang letterbox 640
  (model semantik → instance via connected components). Metrik native Ultralytics juga dicatat (`ultra_*`).
* **Teacher KD per (seed, fold)** = model terbaik Part 1/2/3 yang dilatih di split yang *sama*
  (tanpa leakage antar-fold), dipilih dengan **val** mAP50-95 mask agar test tetap unseen
  (ganti `TEACHER_SELECT_METRIC` kalau ingin test). Student KD memakai imgsz/epoch/batch yang
  sama dengan baseline Part 3 → selisih mAP murni efek KD; `kd_significance` berisi paired t-test,
  Wilcoxon, dan Cohen's dz.
* **Hemat disk**: cache `.npy` uint8 berisi byte JPEG (~10× lebih kecil dari array mentah); dataset
  YOLO & cache fold dibuat sementara lalu dihapus; `last.pt`/`epoch*.pt` dihapus di callback tiap epoch;
  bobot model non-YOLO disimpan fp16 hanya saat val naik. Orchestrator loop (seed, fold) di luar
  sehingga hanya 1 set bobot kerja ada di disk. Retensi bobot final (`RETENTION`): per_fold (1 file per
  model × fold, terbaik lintas seed) atau per_model. Disk < 4 GB → bobot Tier 2 dihapus & Part 2/5 ditunda;
  < 0.5 GB → berhenti dengan state tersimpan (exit 3).
* **Excel incremental**: tiap baris ditulis ke journal `state/journal/*.jsonl` (append, crash-safe) lalu
  `results/part*.xlsx` ditulis ulang secara atomik setelah **setiap** run; history per-epoch ke
  `results/history/*_history.xlsx` tiap 10 run.

## Estimasi waktu (penting)

Spesifikasi penuh = 50 run YOLO-x + 125 run Transformer/CNN + 150 run edge + **675 run KD**
(5 student × 9 α,T × 25) + ±750 unit Part 5. Dengan dataset ratusan–ribuan gambar ini bisa jauh
melebihi 2–4 hari. Opsi mempercepat tanpa mengubah kode:

* `GUAVA_PROFILE=fast` — epoch lebih pendek + `KD_GRID_SCOPE="search_then_full"` (grid α,T hanya di
  seed pertama, seed lain memakai α,T terbaik per student).
* `GUAVA_SEEDS=42,123,2024` / `GUAVA_FOLDS=0,1,2` — subset seed/fold.
* Karena semuanya resume-able, bisa dijalankan bertahap.
