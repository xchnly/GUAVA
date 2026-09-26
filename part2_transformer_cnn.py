"""
part2_transformer_cnn.py — FASE 2: arsitektur non-YOLO (Transformer-seg + CNN-seg).

2A. Transformer-seg : SegFormer-B5, SegFormer-B2 (HF, head diganti num_labels = nc+1)
2B. CNN-seg         : MobileNetV3-L + DeepLabV3+, MobileNetV2 + UNet, EfficientNet-B0 + UNet (SMP)
Model yang gagal preflight (build/bobot/forward) di-skip otomatis; minimal 1 per family.

Loss CE + Dice, AMP bf16, gradient checkpointing (transformer). Bobot HANYA disimpan saat
val mAP50-95_mask naik (1 file ditimpa). Eval val+test → results/part2_transformer_cnn.xlsx.

Jalankan: python part2_transformer_cnn.py [--seed 42] [--fold 0] [--models segformer-b5]
"""
from guava_core import C, parse_part_args, part_main

PART = "part2"


def main():
    import guava_train as T
    args = parse_part_args(__doc__)
    T.baseline_part(PART, "part2_transformer_cnn", [
        ("transformer", C.TRANSFORMER_MODELS, C.EPOCHS_TRANSFORMER, C.BATCH_TRANSFORMER, C.LR["transformer"]),
        ("cnn", C.CNN_MODELS, C.EPOCHS_CNN, C.BATCH_CNN, C.LR["cnn"]),
    ], args, keep_work=False, teacher_candidate=True)


if __name__ == "__main__":
    part_main(PART, main)
