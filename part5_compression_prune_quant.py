"""
part5_compression_prune_quant.py — FASE 5: Kompresi #2 Pruning & #3 Quantization.

Sumber bobot per (seed, fold, student): C.P5_SOURCES
  baseline → weights/work/s_f/part3_{student}.pt (Part 3)
  kd       → weights/work/s_f/kd_{student}.pt  (KD terbaik Part 4)

5A Pruning  : structured channel pruning (torch-pruning, magnitude L2) ratio C.PRUNE_RATIOS,
              fine-tune C.PRUNE_FT_EPOCHS epoch LR kecil. Bandingkan dgn sebelum prune (ref_*).
              → results/part5_pruning.xlsx
5B Quant    : PTQ INT8 static (onnxruntime QDQ, per-channel) + OpenVINO/NNCF INT8 (jika ada);
              QAT ringan (torch.ao FX) untuk C.QAT_N_MODELS student terbaik (by val Part 3).
              Ukur size (MB), latency CPU (ms), mAP. → results/part5_quantization.xlsx
Kombinasi (model × teknik) yang gagal preflight (mis. ShuffleNet tidak bisa di-prune,
EfficientNet-Lite tidak bisa di-trace FX untuk QAT) otomatis di-skip & dicatat di log.

Jalankan: python part5_compression_prune_quant.py [--seed 42] [--fold 0]
"""
import copy
import time

from guava_core import (C, DG, PartState, ResultBook, ensure_deps, env_report, file_mb, finalize_fold, fold_name,
                        load_results, log, parse_part_args, part_main, pick_imgsz, promote_final, read_json,
                        run_unit, unit_key, work_dir, write_json, write_part_done)

PART = "part5"
METRICS = ["mAP50_box", "mAP50-95_box", "mAP50_mask", "mAP50-95_mask", "precision", "recall"]
CAP_PATH = C.STATE_DIR / "compression_preflight.json"


# =============================================================================
# Pruning
# =============================================================================
def prune_model(model, ratio, S, nc):
    import torch
    import torch_pruning as tp
    model = copy.deepcopy(model).cpu().float().eval()
    ex = torch.rand(1, 3, S, S) * 255
    ign = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d) and m.out_channels == nc + 1]
    pr = tp.pruner.MetaPruner(model, ex, importance=tp.importance.MagnitudeImportance(p=2),
                              pruning_ratio=ratio, ignored_layers=ign)
    pr.step()
    with torch.no_grad():
        y = model(ex)
    assert y.shape[1] == nc + 1
    return model


# =============================================================================
# QAT (FX) — hanya bagian `net`; normalisasi & resize tetap float di SegNet
# =============================================================================
def qat_prepare(model, S):
    import torch
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx
    torch.backends.quantized.engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"
    q = copy.deepcopy(model).cpu().float().train()
    q.net = prepare_qat_fx(q.net, get_default_qat_qconfig_mapping(torch.backends.quantized.engine),
                           (torch.rand(1, 3, S, S),))
    return q


def qat_convert(q):
    from torch.ao.quantization.quantize_fx import convert_fx
    q = q.cpu().eval()
    q.net = convert_fx(q.net)
    return q


def capabilities(students, nc):
    """Preflight per teknik (di-cache): prune / qat / ort / openvino."""
    import torch
    import guava_models as M
    cap = read_json(CAP_PATH, {})
    for s in students:
        if s in cap:
            continue
        c = {}
        m = M.build(s, nc, pretrained=False)
        for tech, test in (("prune", lambda: prune_model(m, 0.4, 128, nc)),
                           ("qat", lambda: qat_convert(qat_prepare(m, 128))(torch.rand(1, 3, 128, 128) * 255))):
            try:
                test()
                c[tech] = True
            except Exception as e:
                c[tech] = False
                log.warning(f"[P5] {s}: {tech} tidak didukung ({repr(e)[:150]}) → di-skip")
        cap[s] = c
        write_json(CAP_PATH, cap)
    try:
        import nncf  # noqa: F401
        import openvino  # noqa: F401
        ov = True
    except Exception:
        ov = False
    return cap, ov


# =============================================================================
# ONNX / ORT / OpenVINO
# =============================================================================
def export_onnx(model, S, path, opset=13):
    import torch
    m = copy.deepcopy(model).cpu().float().eval()
    m.capture_feat = False
    x = torch.rand(1, 3, S, S) * 255
    kw = dict(input_names=["images"], output_names=["logits"], opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(m, x, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, x, str(path), **kw)
    return path


class _Calib:
    def __init__(self, items, S, n):
        import guava_data as D
        import numpy as np
        self.data = [{"images": D.letterbox(D.load_image(it), S).transpose(2, 0, 1)[None].astype(np.float32)}
                     for it in items[:n]]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return self.data[self.i - 1]

    def rewind(self):
        self.i = 0


def ort_ptq(fp32, int8, calib_items, S):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = fp32.with_suffix(".pre.onnx")
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception:
        src = fp32
    quantize_static(str(src), str(int8), _Calib(calib_items, S, C.CALIB_IMAGES), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    return int8


class OrtRunner:
    def __init__(self, path, nc):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = C.LAT_CPU_THREADS
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.nc = nc

    def __call__(self, batch_u8):
        import numpy as np
        return np.concatenate([self.sess.run(None, {"images": b[None].astype(np.float32)})[0] for b in batch_u8])


class OVRunner:
    def __init__(self, compiled, nc):
        self.c, self.nc = compiled, nc

    def __call__(self, batch_u8):
        import numpy as np
        return np.concatenate([self.c(b[None].astype(np.float32))[0] for b in batch_u8])


def ov_ptq(fp32, calib_items, S, nc):
    import nncf
    import openvino as ov
    core = ov.Core()
    model = core.read_model(str(fp32))
    cal = _Calib(calib_items, S, C.CALIB_IMAGES)
    q = nncf.quantize(model, nncf.Dataset([d["images"] for d in cal.data]), subset_size=len(cal.data))
    xml = fp32.with_suffix(".ov_int8.xml")
    ov.save_model(q, str(xml))
    comp = core.compile_model(q, "CPU", {"INFERENCE_NUM_THREADS": C.LAT_CPU_THREADS})
    return OVRunner(comp, nc), xml


def runner_latency(fn, S):
    import numpy as np
    x = (np.random.rand(1, 3, S, S) * 255).astype(np.uint8)
    for _ in range(C.LAT_WARMUP):
        fn(x)
    ts = []
    for _ in range(C.LAT_RUNS):
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return sorted(ts)[len(ts) // 2]


# =============================================================================
def main():
    args = parse_part_args(__doc__)
    ensure_deps()
    env_report()
    DG.report_disk()
    DG.guard_before_training()

    import guava_data as D
    import guava_models as M
    import guava_train as T

    master = D.load_master()
    nc = master["nc"]
    students = M.usable(args.model_list or C.P5_MODELS, nc, family="edge")
    cap, has_ov = capabilities(students, nc)
    state = PartState(PART)

    # QAT: pilih sekali (disimpan di state) → konsisten lintas fold
    qat_models = state.get("qat_models")
    if not qat_models:
        cands = [s for s in students if cap[s].get("qat")]
        p3 = load_results("part3_edge_baseline")
        if not p3.empty:
            rank = p3[p3["model"].isin(cands) & (p3["status"] == "OK")].groupby("model")["val_mAP50-95_mask"].mean()
            cands = sorted(cands, key=lambda s: -rank.get(s, -1))
        qat_models = cands[:C.QAT_N_MODELS]
        state.set("qat_models", qat_models)
    log.info(f"[P5] student={students} | QAT={qat_models} | OpenVINO/NNCF={'ya' if has_ov else 'tidak'}")

    book_p = ResultBook(C.EXCEL_DIR / "part5_pruning.xlsx")
    book_q = ResultBook(C.EXCEL_DIR / "part5_quantization.xlsx")
    tokens = sum((M.cache_tokens(s) for s in students), [])

    def plan(seed, fold):
        out = []
        for s in students:
            for src in C.P5_SOURCES:
                if cap[s].get("prune"):
                    out += [("prune", s, src, r) for r in C.PRUNE_RATIOS]
                out.append(("ptq_int8_ort", s, src, None))
                if has_ov:
                    out.append(("ptq_int8_ov", s, src, None))
            if s in qat_models:
                out.append(("qat_int8", s, "baseline", None))
        return out

    def key_of(tech, s, src, r, seed, fold):
        return unit_key(s, seed, fold, tech, f"{src}" + (f"_r{r}" if r is not None else ""))

    expected = []
    for seed in C.SEEDS:
        for fold in C.FOLDS:
            expected += [key_of(*p, seed, fold) for p in plan(seed, fold)]

    for seed in args.seeds:
        for fold in args.folds:
            units = plan(seed, fold)
            if all(state.is_done(key_of(*u, seed, fold)) for u in units):
                continue
            wd = work_dir(seed, fold)
            fd = D.prepare_fold(seed, fold, master)
            ref_cache = {}

            def source(s, src):
                p = wd / (f"part3_{s}.pt" if src == "baseline" else f"kd_{s}.pt")
                if not p.exists():
                    return None, None, None
                if (s, src) not in ref_cache:
                    m, meta = M.load_model(p, T.dev())
                    S = pick_imgsz(seed, fold, s)
                    ref = {**T.evaluate_full(m, fd, S), **T.profile_semseg(m, S, p)}
                    ref_cache[(s, src)] = {f"ref_{k}": v for k, v in ref.items()}
                    m.cpu()
                    ref_cache[(s, src, "m")] = m
                return ref_cache[(s, src, "m")], ref_cache[(s, src)], p

            for tech, s, src, r in units:
                key = key_of(tech, s, src, r, seed, fold)
                if state.is_done(key):
                    continue
                model, ref, spath = source(s, src)
                if model is None:
                    log.warning(f"[P5] {key}: bobot sumber '{src}' belum ada (Part 3/4 belum jalan) → skip")
                    continue
                S = pick_imgsz(seed, fold, s)
                base = {"part": PART, "model": s, "family": "edge", "seed": seed, "fold": fold, "imgsz": S,
                        "technique": tech if r is None else "prune", "hyperparam": f"{src}" + (f"_r{r}" if r else ""),
                        "source": src, "prune_ratio": r, **ref}
                book = book_p if tech == "prune" else book_q

                def delta(res):
                    for sp in ("val", "test"):
                        for m in METRICS:
                            if f"{sp}_{m}" in res and f"ref_{sp}_{m}" in ref:
                                res[f"delta_{sp}_{m}"] = res[f"{sp}_{m}"] - ref[f"ref_{sp}_{m}"]
                    return res

                if tech == "prune":
                    def fn(attempt, s=s, r=r, S=S, key=key, base=base, model=model, src=src):
                        pruned = prune_model(model, r, S, nc)
                        wpath = wd / f"prune{r}_{src}_{s}.pt"
                        _, res = T.semantic_unit(book, key, base, attempt, fd, S, C.PRUNE_FT_EPOCHS, C.BATCH_EDGE,
                                                 C.PRUNE_FT_LR, seed, wpath,
                                                 make_model=lambda: copy.deepcopy(pruned), save_full=True)
                        promote_final(wpath, "prune", f"prune{r}_{src}_{s}", seed, fold, res["val_mAP50-95_mask"],
                                      {"imgsz": S, "ratio": r, "source": src})
                        wpath.unlink(missing_ok=True)
                        return delta(res)
                elif tech in ("ptq_int8_ort", "ptq_int8_ov"):
                    def fn(attempt, s=s, S=S, model=model, tech=tech):
                        fp32 = C.TMP_DIR / f"{s}_{fold_name(seed, fold)}.onnx"
                        export_onnx(model, S, fp32)
                        try:
                            if tech == "ptq_int8_ort":
                                q = ort_ptq(fp32, fp32.with_suffix(".int8.onnx"), fd.train_orig, S)
                                runner, size = OrtRunner(q, nc), file_mb(q)
                            else:
                                runner, xml = ov_ptq(fp32, fd.train_orig, S, nc)
                                size = file_mb(xml) + file_mb(xml.with_suffix(".bin"))
                            res = {"size_MB": round(size, 2), "fp32_onnx_MB": round(file_mb(fp32), 2)}
                            res.update(T.prefix(T.eval_semantic(runner, fd.val, S, batch=8), "val"))
                            res.update(T.prefix(T.eval_semantic(runner, fd.test, S, batch=8), "test"))
                            res["latency_cpu_ms"] = res["latency_ms"] = round(runner_latency(runner, S), 2)
                            res["params_M"], res["flops_G"] = ref.get("ref_params_M"), ref.get("ref_flops_G")
                            return delta(res)
                        finally:
                            for f in C.TMP_DIR.glob(f"{s}_{fold_name(seed, fold)}*"):
                                f.unlink(missing_ok=True)
                else:  # qat_int8
                    def fn(attempt, s=s, S=S, key=key, base=base, model=model):
                        import torch
                        holder = {}

                        def mk():
                            q = qat_prepare(model, S)
                            q.nc, q.name = model.nc, model.name
                            holder["q"] = q
                            return q
                        hist_base = {"part": PART, "model": s, "seed": seed, "fold": fold, "technique": "qat_int8"}
                        info, used = T.with_oom_retry(lambda b: T.train_semantic(
                            mk(), fd, S, C.QAT_EPOCHS, b, C.QAT_LR, seed, save_path=None, amp=False,
                            history=lambda row: book.add("history", {"unit": key, "attempt": attempt, **hist_base,
                                                                     **row}, flush=False)), C.BATCH_EDGE)
                        qm = qat_convert(holder["q"])
                        qm.nc = model.nc
                        tmp = C.TMP_DIR / f"qat_{s}.pt"
                        torch.save(qm.state_dict(), tmp)
                        res = {"size_MB": round(file_mb(tmp), 2), "best_epoch": info["best_epoch"],
                               "epochs_run": info["epochs_run"], "batch": used}
                        tmp.unlink(missing_ok=True)
                        res.update(T.prefix(T.eval_semantic(qm, fd.val, S, batch=8), "val"))
                        res.update(T.prefix(T.eval_semantic(qm, fd.test, S, batch=8), "test"))
                        from guava_core import measure_latency
                        res["latency_cpu_ms"] = res["latency_ms"] = round(measure_latency(qm, S, "cpu"), 2)
                        res["params_M"], res["flops_G"] = ref.get("ref_params_M"), ref.get("ref_flops_G")
                        return delta(res)

                run_unit(state, book, key, base, fn, keep_models=tokens, keep_folds=[fd.name])
                DG.clean_all(keep_models=tokens, keep_folds=[fd.name], verbose=False)
            book_p.flush(include_history=True)
            book_q.flush(include_history=True)
            if all(state.is_done(key_of(*u, seed, fold)) for u in units):
                finalize_fold(seed, fold)   # Part 5 = konsumen terakhir bobot kerja (s,f)

    book_p.flush(include_history=True)
    book_q.flush(include_history=True)
    if not args.no_final:
        write_part_done(PART, state, expected, {"students": students, "qat_models": qat_models})
        DG.clean_all(keep_models=[] if DG.disk_free_gb() < C.CACHE_PURGE_FREE_GB else tokens, keep_folds=[])
        DG.report_disk()


if __name__ == "__main__":
    part_main(PART, main)
