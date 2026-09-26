"""
part3_edge_candidates.py — FASE 3: kandidat edge/mobile (non-YOLO, 6 model; ≥5 wajib).

Kandidat (C.EDGE_MODELS): MobileNetV3-Small + LR-ASPP, EfficientNet-Lite0 + UNet,
ShuffleNetV2 + UNet, MobileOne-S0 + UNet, + model ringan Part 2 (MobileNetV2-UNet,
MobileNetV3-L DeepLabV3+) yang dilatih ulang dengan profil edge (EPOCHS_EDGE, BATCH_EDGE).

Baseline per (seed, fold): params, FLOPs (thop), ukuran file (MB), latency CPU & GPU (batch=1),
mAP50 / mAP50-95 box & mask val/test → results/part3_edge_baseline.xlsx (incremental).
Bobot tetap di weights/work/s{seed}_f{fold}/ untuk Part 4 (pembanding KD) & Part 5 (prune/quant).

Jalankan: python part3_edge_candidates.py [--seed 42] [--fold 0]
"""
from guava_core import C, parse_part_args, part_main

PART = "part3"


def main():
    import guava_train as T
    args = parse_part_args(__doc__)
    T.baseline_part(PART, "part3_edge_baseline", [
        ("edge", C.EDGE_MODELS, C.EPOCHS_EDGE, C.BATCH_EDGE, C.LR["edge"]),
    ], args, keep_work=True, teacher_candidate="part3" in C.TEACHER_PARTS)


if __name__ == "__main__":
    part_main(PART, main)
