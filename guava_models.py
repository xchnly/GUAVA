"""
guava_models.py — Registry model + builder + save/load + PREFLIGHT otomatis.

Semua model non-YOLO dibungkus SegNet:
  input  : float/uint8 (B,3,S,S) RGB 0..255   (normalisasi ImageNet di dalam model → ONNX-friendly)
  output : logits (B, nc+1, S, S)             (kanal 0 = background)
  .feat  : feature map input classifier (untuk feature-KD), ditangkap via forward-pre-hook.

Preflight: tiap model dicoba build (dengan bobot pretrained) + forward/backward kecil.
Yang gagal di-skip otomatis; minimal 1 model per family wajib lolos.
"""
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from guava_core import C, log, read_json, write_json

NO_PRETRAINED = os.environ.get("GUAVA_NO_PRETRAINED") == "1"   # HANYA untuk uji offline

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name → (kind, family default, spec)
REGISTRY = {
    # --- YOLO (ultralytics, instance seg) ---
    "yolo11x-seg": ("yolo", "yolo", {"weights": "yolo11x-seg.pt"}),
    "yolo26x-seg": ("yolo", "yolo", {"weights": "yolo26x-seg.pt"}),
    "yolo11n-seg": ("yolo", "yolo", {"weights": "yolo11n-seg.pt"}),   # uji cepat
    "yolo26n-seg": ("yolo", "yolo", {"weights": "yolo26n-seg.pt"}),
    # --- Transformer-seg (HF SegFormer) ---
    # repo dicoba berurutan; B5 versi ADE dirilis di 640×640 (tidak ada varian 512)
    "segformer-b5": ("segformer", "transformer", {"repos": ["nvidia/segformer-b5-finetuned-ade-640-640",
                                                            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                                                            "nvidia/mit-b5"]}),
    "segformer-b2": ("segformer", "transformer", {"repos": ["nvidia/segformer-b2-finetuned-ade-512-512",
                                                            "nvidia/mit-b2"]}),
    # --- CNN-seg non-YOLO (SMP + encoder timm) ---
    "mnv3L-deeplabv3plus": ("smp", "cnn", {"arch": "DeepLabV3Plus", "enc": "tu-mobilenetv3_large_100"}),
    "mnv2-unet": ("smp", "cnn", {"arch": "Unet", "enc": "tu-mobilenetv2_100"}),
    "effb0-unet": ("smp", "cnn", {"arch": "Unet", "enc": "tu-efficientnet_b0"}),
    # --- Edge/mobile ---
    "efflite0-unet": ("smp", "edge", {"arch": "Unet", "enc": "tu-tf_efficientnet_lite0"}),
    "mobileone-s0-unet": ("smp", "edge", {"arch": "Unet", "enc": "tu-mobileone_s0"}),
    "mnv3S-lraspp": ("lraspp_small", "edge", {}),
    "shufflenetv2-unet": ("shufflenet_unet", "edge", {}),
}

# Token cache (HF repo / nama file) per model → dipakai clean_all(keep_models=...)
def cache_tokens(name):
    kind, _, spec = REGISTRY[name]
    return [name, *spec.get("repos", []), spec.get("weights", ""),
            spec.get("enc", "").replace("tu-", ""), {"lraspp_small": "mobilenet_v3_small",
                                                      "shufflenet_unet": "shufflenetv2"}.get(kind, "")]


def kind_of(name):
    return REGISTRY[name][0]


def is_yolo(name):
    return kind_of(name) == "yolo"


# =============================================================================
# Wrapper
# =============================================================================
class SegNet(nn.Module):
    def __init__(self, net, kind, name, nc, feat_module=None):
        super().__init__()
        self.net, self.kind, self.name, self.nc = net, kind, name, nc
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)
        self.feat = None
        self.capture_feat = False
        if feat_module is not None:
            feat_module.register_forward_pre_hook(self._hook)

    def _hook(self, mod, inp):
        if self.capture_feat:
            self.feat = inp[0]

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x.float() - self.mean) / self.std
        y = self.net(x)
        if hasattr(y, "logits"):
            y = y.logits
        elif isinstance(y, dict):
            y = y["out"]
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


class LiteUNet(nn.Module):
    """Decoder UNet ringan untuk backbone torchvision (ShuffleNetV2)."""

    def __init__(self, body, chs, n_out, dec=(96, 64, 48, 32)):
        super().__init__()
        self.body = body  # create_feature_extractor → dict fitur (resolusi naik → turun)
        self.keys = list(chs)
        c = list(chs.values())[::-1]  # dalam → dangkal
        blocks, cin = [], c[0]
        for skip, cout in zip(c[1:], dec):
            blocks.append(nn.Sequential(nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(cin, n_out, 1)

    def forward(self, x):
        f = self.body(x)
        feats = [f[k] for k in self.keys][::-1]
        y = feats[0]
        for blk, skip in zip(self.blocks, feats[1:]):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        return self.head(y)


# =============================================================================
# Builder
# =============================================================================
def build(name, nc, pretrained=True):
    """Return SegNet (non-YOLO). YOLO dibangun lewat guava_train (ultralytics)."""
    kind, _, spec = REGISTRY[name]
    K = nc + 1
    pretrained = pretrained and not NO_PRETRAINED
    if kind == "smp":
        import segmentation_models_pytorch as smp
        net = getattr(smp, spec["arch"])(spec["enc"], encoder_weights="imagenet" if pretrained else None, classes=K)
        return SegNet(net, kind, name, nc, net.segmentation_head)
    if kind == "segformer":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            net, errs = None, []
            for repo in spec["repos"]:
                try:
                    net = SegformerForSemanticSegmentation.from_pretrained(repo, num_labels=K,
                                                                           ignore_mismatched_sizes=True)
                    break
                except Exception as e:
                    errs.append(f"{repo}: {repr(e)[:150]}")
            if net is None:
                raise OSError(" | ".join(errs))
            # catat arsitektur yang benar-benar dipakai → load ulang bobot selalu cocok
            arch = read_json(ARCH_PATH, {})
            arch[name] = {"repo": repo, "config": {k: v for k, v in net.config.to_dict().items()
                                                   if k not in ("id2label", "label2id", "num_labels")}}
            write_json(ARCH_PATH, arch)
        else:
            saved = read_json(ARCH_PATH, {}).get(name)
            cfg = SegformerConfig(**saved["config"]) if saved else _segformer_cfg(name)
            cfg.num_labels = K
            net = SegformerForSemanticSegmentation(cfg)
        if C.GRAD_CKPT_TRANSFORMER:
            try:
                net.gradient_checkpointing_enable()
            except Exception:
                pass
        return SegNet(net, kind, name, nc, net.decode_head.classifier)
    if kind == "lraspp_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.models.segmentation.lraspp import LRASPP
        bb = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
                                dilated=True).features
        idx = [0] + [i for i, b in enumerate(bb) if getattr(b, "_is_cn", False)] + [len(bb) - 1]
        lo, hi = idx[-4], idx[-1]
        body = IntermediateLayerGetter(bb, {str(lo): "low", str(hi): "high"})
        net = LRASPP(body, bb[lo].out_channels, bb[hi].out_channels, K)
        return SegNet(net, kind, name, nc, net.classifier.high_classifier)
    if kind == "shufflenet_unet":
        from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        bb = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        nodes = {"conv1": "s2", "maxpool": "s4", "stage2": "s8", "stage3": "s16", "conv5": "s32"}
        body = create_feature_extractor(bb, nodes)
        chs = {"s2": 24, "s4": 24, "s8": 116, "s16": 232, "s32": 1024}
        net = LiteUNet(body, chs, K)
        return SegNet(net, kind, name, nc, net.head)
    raise ValueError(f"model tidak dikenal: {name}")


def _segformer_cfg(name):
    from transformers import SegformerConfig
    if name.endswith("b5"):
        return SegformerConfig(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768)
    return SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768)


# =============================================================================
# Save / load
# =============================================================================
def save_model(model, path, meta=None, full=False):
    """fp16 state_dict (hemat disk ½). full=True → pickle modul utuh (model hasil pruning)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if full:
        m = copy.deepcopy(model).cpu().half()
        torch.save({"full_model": m, "name": model.name, "nc": model.nc, "meta": meta or {}}, tmp)
    else:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
        torch.save({"state_dict": sd, "name": model.name, "nc": model.nc, "meta": meta or {}}, tmp)
    tmp.replace(path)
    return path


def load_model(path, dev="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "full_model" in ck:
        m = ck["full_model"].float()
        # re-register hook fitur (hilang saat pickle? hook tetap tersimpan; aman untuk dipasang ulang)
    else:
        m = build(ck["name"], ck["nc"], pretrained=False)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return m.to(dev).eval(), ck.get("meta", {})


# =============================================================================
# PREFLIGHT
# =============================================================================
PREFLIGHT_PATH = C.STATE_DIR / "model_preflight.json"
ARCH_PATH = C.STATE_DIR / "model_arch.json"
FAMILY_LISTS = {"yolo": "YOLO_MODELS", "transformer": "TRANSFORMER_MODELS", "cnn": "CNN_MODELS",
                "edge": "EDGE_MODELS"}


def _check_one(name, nc):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if is_yolo(name):
        from guava_train import yolo_pretrained
        from ultralytics import YOLO
        m = YOLO(str(yolo_pretrained(name)))
        m.model.to(dev).train()
        m.model(torch.rand(1, 3, 128, 128, device=dev))
        del m
        return True
    m = build(name, nc, pretrained=True).to(dev).train()
    m.capture_feat = True
    y = m(torch.rand(2, 3, 128, 128, device=dev) * 255)
    assert y.shape == (2, nc + 1, 128, 128), f"shape output salah {tuple(y.shape)}"
    assert m.feat is not None, "hook fitur KD tidak menangkap apa-apa"
    y.float().mean().backward()
    del m, y
    return True


def preflight(names, nc, force=False):
    """Uji build + pretrained + forward/backward. Hasil di-cache di state/model_preflight.json."""
    res = read_json(PREFLIGHT_PATH, {})
    for n in names:
        sig = json.dumps(REGISTRY[n][2], sort_keys=True)   # spesifikasi berubah → uji ulang
        if n in res and res[n]["ok"] and not force:
            continue
        if n in res and not res[n]["ok"] and res[n].get("tries", 0) >= 2 and res[n].get("sig") == sig \
                and not force:
            continue
        if n in res and res[n].get("sig") != sig:
            res[n]["tries"] = 0
        try:
            _check_one(n, nc)
            res[n] = {"ok": True, "sig": sig}
            log.info(f"[PREFLIGHT] ✔ {n}")
        except Exception as e:
            res[n] = {"ok": False, "err": repr(e)[:600], "tries": res.get(n, {}).get("tries", 0) + 1, "sig": sig}
            log.warning(f"[PREFLIGHT] ✘ {n}: {repr(e)[:300]} → di-skip")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_json(PREFLIGHT_PATH, res)
    return res


def usable(names, nc, family=None, required=True):
    """Filter model yang lolos preflight. required → raise kalau family tidak punya model tersisa."""
    names = [n for n in names if n in REGISTRY]
    res = preflight(names, nc)
    ok = [n for n in names if res.get(n, {}).get("ok")]
    skipped = [n for n in names if n not in ok]
    if skipped:
        log.warning(f"[MODELS] di-skip (gagal preflight): {skipped}")
    if required and not ok:
        raise RuntimeError(f"Tidak ada model {family or ''} yang lolos preflight dari {names}. "
                           f"Cek koneksi (download bobot) / versi library. Detail: {PREFLIGHT_PATH}")
    return ok


def require_all_families(nc):
    """Pastikan tiap family (yolo, transformer, cnn, edge) punya ≥1 model yang bisa dipakai."""
    out = {}
    for fam, attr in FAMILY_LISTS.items():
        out[fam] = usable(getattr(C, attr), nc, family=fam, required=True)
    log.info(f"[MODELS] dipakai: {out}")
    return out
