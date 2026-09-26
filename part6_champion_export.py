"""
part6_champion_export.py — FASE 6: pilih champion student edge (non-YOLO) + export.

Kandidat = konfigurasi (model, teknik, hyperparam) dari Part 3 (baseline), Part 4 (KD),
Part 5 (pruning / PTQ / QAT), dirata-rata lintas seed × fold (test set):

    score = w1·(1 − mAP_drop) + w2·(1 − latency_norm) + w3·(1 − size_norm)     (w = 0.6, 0.2, 0.2)
    mAP_drop     = max(0, (mAP_teacher − mAP) / mAP_teacher)   [mAP50-95 mask, test]
    latency_norm = latency_cpu / max(latency_cpu kandidat)
    size_norm    = size_MB / max(size_MB kandidat)

Export champion → artifacts/CHAMPION_EDGE/:
    champion.pt, champion_fp32.onnx (opset 12, simplified), champion_int8.onnx (ORT PTQ),
    openvino/ (IR FP16), champion_fp16.engine (TensorRT, bila tersedia), TFLite (opsional),
    champion_info.json (preprocessing: RGB float 0..255 NCHW, output logits nc+1 kanal).
ONNX/engine intermediate lain dihapus.
"""
import shutil

from guava_core import (C, DG, PartState, ResultBook, ensure_deps, env_report, file_mb, find_final, load_results,
                        log, now, parse_part_args, part_main, read_json, write_json, write_part_done)

PART = "part6"
OUT = C.ARTIFACTS / "CHAMPION_EDGE"
KEY = "test_mAP50-95_mask"


def candidates():
    import pandas as pd
    frames = []
    for stem, part in (("part3_edge_baseline", "part3"), ("part4_kd", "part4"), ("part5_pruning", "part5"),
                       ("part5_quantization", "part5")):
        df = load_results(stem)
        if df.empty:
            continue
        df = df[df["status"] == "OK"].copy()
        df["part"] = part
        frames.append(df)
    if not frames:
        raise RuntimeError("Belum ada hasil Part 3/4/5")
    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby(["part", "model", "technique", "hyperparam"]).agg(
        n_runs=(KEY, "size"), test_mAP50_95_mask=(KEY, "mean"), test_mAP50_95_mask_std=(KEY, "std"),
        test_mAP50_mask=("test_mAP50_mask", "mean"), val_mAP50_95_mask=("val_mAP50-95_mask", "mean"),
        latency_cpu_ms=("latency_cpu_ms", "mean"), size_MB=("size_MB", "mean"),
        params_M=("params_M", "mean"), flops_G=("flops_G", "mean")).reset_index()
    return agg, df


def teacher_ref(rows):
    if "teacher_test_mAP50-95_mask" in rows.columns and rows["teacher_test_mAP50-95_mask"].notna().any():
        return float(rows["teacher_test_mAP50-95_mask"].mean())
    for stem in ("part1_yolo", "part2_transformer_cnn"):
        df = load_results(stem)
        if not df.empty:
            return float(df[df["status"] == "OK"].groupby("model")[KEY].mean().max())
    return float(rows[KEY].max())


def score(agg, ref):
    w1, w2, w3 = C.CHAMP_W
    a = agg.copy()
    if not ref or ref != ref or ref <= 0:   # teacher belum terlatih / NaN → pakai kandidat terbaik
        ref = max(float(a["test_mAP50_95_mask"].max()), 1e-9)
    a["teacher_ref_mAP50_95_mask"] = ref
    a["mAP_drop"] = ((ref - a["test_mAP50_95_mask"]) / ref).clip(lower=0)
    a["latency_norm"] = a["latency_cpu_ms"] / a["latency_cpu_ms"].max()
    a["size_norm"] = a["size_MB"] / a["size_MB"].max()
    a["score"] = (w1 * (1 - a["mAP_drop"]) + w2 * (1 - a["latency_norm"]) + w3 * (1 - a["size_norm"])).fillna(-1)
    # konfigurasi tanpa cakupan penuh seed×fold dihukum ringan agar tidak menang karena kebetulan
    full = len(C.SEEDS) * len(C.FOLDS)
    a["coverage"] = a["n_runs"] / full
    return a.sort_values(["score", "coverage"], ascending=False).reset_index(drop=True)


def resolve_weights(row):
    """(path, family, key) bobot float untuk konfigurasi champion."""
    m, tech, hp = row["model"], row["technique"], row["hyperparam"]
    if tech == "baseline":
        return find_final("edge", m)[0], "edge"
    if tech == "kd":
        return find_final("kd", f"{m}_{hp}")[0], "kd"
    if tech == "prune":
        src, r = hp.split("_r")
        return find_final("prune", f"prune{r}_{src}_{m}")[0], "prune"
    src = hp.split("_")[0]  # ptq_* / qat: bobot float sumber
    if src == "kd":
        best = None
        for d in (C.WEIGHTS_DIR / "best" / "kd").glob(f"{m}_a*"):
            idx = read_json(d / "index.json", {})
            for fn, v in idx.items():
                if best is None or v["score"] > best[1]:
                    best = (d / fn, v["score"])
        if best:
            return best[0], "kd"
    return find_final("edge", m)[0], "edge"


def export_all(model, S, nc, names, row, wpath):
    import torch
    import guava_data as D
    from part5_compression_prune_quant import OrtRunner, export_onnx, ort_ptq, runner_latency
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copy2(wpath, OUT / "champion.pt")
    info = {"created": now(), "model": row["model"], "technique": row["technique"], "hyperparam": row["hyperparam"],
            "imgsz": S, "nc": nc, "class_names": names, "input": "float32 RGB 0..255 NCHW (1,3,S,S), letterbox "
            "top-left pad=114", "output": f"logits (1,{nc + 1},S,S), kanal 0 = background, argmax → kelas-1",
            "score_row": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in row.items()},
            "exports": {}}
    m = model.cpu().float().eval()
    m.capture_feat = False
    x = torch.rand(1, 3, S, S) * 255
    onnx_p = OUT / "champion_fp32.onnx"
    for opset in (C.ONNX_OPSET, 13, 17):
        try:
            kw = dict(input_names=["images"], output_names=["logits"], opset_version=opset, do_constant_folding=True)
            try:
                torch.onnx.export(m, x, str(onnx_p), dynamo=False, **kw)
            except TypeError:
                torch.onnx.export(m, x, str(onnx_p), **kw)
            info["exports"]["onnx_opset"] = opset
            break
        except Exception as e:
            log.warning(f"[EXPORT] ONNX opset {opset} gagal: {repr(e)[:200]}")
    if onnx_p.exists():
        try:
            import onnx
            try:
                import onnxsim
                sm, ok = onnxsim.simplify(onnx.load(str(onnx_p)))
            except ImportError:
                import onnxslim
                sm, ok = onnxslim.slim(onnx.load(str(onnx_p))), True
            if ok:
                onnx.save(sm, str(onnx_p))
                info["exports"]["onnx_simplified"] = True
        except Exception as e:
            log.warning(f"[EXPORT] simplify gagal: {e}")
        info["exports"]["onnx_fp32_MB"] = round(file_mb(onnx_p), 2)
        # INT8 ORT (QDQ per-channel butuh opset ≥ 13 → export terpisah di tmp/)
        try:
            fd = D.prepare_fold(C.SEEDS[0], C.FOLDS[0])
            src13 = export_onnx(m, S, C.TMP_DIR / "champion_op13.onnx", opset=13)
            q = ort_ptq(src13, OUT / "champion_int8.onnx", fd.train_orig, S)
            for extra in C.TMP_DIR.glob("champion_op13*"):
                extra.unlink()
            info["exports"]["onnx_int8_MB"] = round(file_mb(q), 2)
            info["exports"]["onnx_int8_latency_cpu_ms"] = round(runner_latency(OrtRunner(q, nc), S), 2)
        except Exception as e:
            log.warning(f"[EXPORT] ORT INT8 gagal: {repr(e)[:200]}")
        # OpenVINO
        try:
            import openvino as ov
            (OUT / "openvino").mkdir(exist_ok=True)
            ov.save_model(ov.convert_model(str(onnx_p)), str(OUT / "openvino" / "champion.xml"), compress_to_fp16=True)
            info["exports"]["openvino_MB"] = round(file_mb(OUT / "openvino"), 2)
        except Exception as e:
            log.warning(f"[EXPORT] OpenVINO gagal: {repr(e)[:200]}")
        # TensorRT
        if C.EXPORT_TENSORRT and torch.cuda.is_available():
            try:
                import tensorrt as trt
                lg = trt.Logger(trt.Logger.WARNING)
                b = trt.Builder(lg)
                try:
                    net = b.create_network(0)
                except Exception:
                    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(net, lg)
                if not parser.parse(onnx_p.read_bytes()):
                    raise RuntimeError(str(parser.get_error(0)))
                cfg = b.create_builder_config()
                cfg.set_flag(trt.BuilderFlag.FP16)
                eng = b.build_serialized_network(net, cfg)
                (OUT / "champion_fp16.engine").write_bytes(bytes(eng))
                info["exports"]["tensorrt_MB"] = round(file_mb(OUT / "champion_fp16.engine"), 2)
            except Exception as e:
                log.warning(f"[EXPORT] TensorRT dilewati: {repr(e)[:200]}")
        if C.EXPORT_TFLITE:
            try:
                import subprocess
                subprocess.run(["onnx2tf", "-i", str(onnx_p), "-o", str(OUT / "tflite")], check=True,
                               capture_output=True)
                info["exports"]["tflite"] = True
            except Exception as e:
                log.warning(f"[EXPORT] TFLite gagal: {repr(e)[:200]}")
    write_json(OUT / "champion_info.json", info)
    return info


def main():
    args = parse_part_args(__doc__)
    ensure_deps()
    env_report()
    import guava_data as D
    import guava_models as M
    agg, rows = candidates()
    ref = teacher_ref(rows)
    ranked = score(agg, ref)
    book = ResultBook(C.EXCEL_DIR / "part6_champion.xlsx")
    for sh in ("candidates", "champion", "exports"):
        (book.jdir / f"part6_champion__{sh}.jsonl").unlink(missing_ok=True)  # selalu dihitung ulang
    book.add("candidates", ranked.to_dict("records"), flush=False)
    master = D.load_master()
    champ, info = None, None
    for _, row in ranked.iterrows():
        wpath, fam = resolve_weights(row)
        if wpath is None:
            log.warning(f"[CHAMP] bobot {row['model']}/{row['technique']}/{row['hyperparam']} tidak ada → berikutnya")
            continue
        model, meta = M.load_model(wpath)
        S = int(meta.get("imgsz", C.IMGSZ_CHOICES[1]))
        log.info(f"[CHAMP] champion = {row['model']} | {row['technique']} | {row['hyperparam']} | score={row['score']:.4f}")
        info = export_all(model, S, master["nc"], master["names"], row.to_dict(), wpath)
        champ = row
        break
    if champ is None:
        raise RuntimeError("Tidak ada kandidat dengan bobot tersedia")
    book.add("champion", {**champ.to_dict(), "weights": str(wpath)}, flush=False)
    book.add("exports", [{"export": k, "value": v} for k, v in info["exports"].items()], flush=True)
    DG.clean_intermediate_exports()
    DG.clean_npy_cache(C.CACHE_DIR, keep_folds=[])   # cache fold kalibrasi
    state = PartState(PART)
    state.mark_done("champion")
    if not args.no_final:
        write_part_done(PART, state, ["champion"], {"champion": f"{champ['model']}|{champ['technique']}|"
                                                                  f"{champ['hyperparam']}"})
    DG.report_disk()


if __name__ == "__main__":
    part_main(PART, main)
