"""
part4_compression_kd.py — FASE 4: Kompresi #1 — Knowledge Distillation (FOKUS RISET).

Teacher : per (seed, fold) otomatis = model terbaik Part 1/2/3 pada split yang SAMA
          (metrik C.TEACHER_SELECT_METRIC, default val mAP50-95 mask → test tetap unseen,
          tanpa leakage antar-fold). Disimpan di weights/work/s{seed}_f{fold}/teacher.pt.
Student : C.KD_STUDENTS (≥5 model edge non-YOLO dari Part 3) yang lolos preflight.

    L = alpha·L_task + beta·L_kd_logits + gamma·L_kd_features
      L_task        = CE + Dice (mask)
      L_kd_logits   = KL(softmax(t/T) ‖ softmax(s/T))·T²
      L_kd_features = MSE feature map teacher vs student (proyeksi 1×1 bila dim beda)

Grid: alpha ∈ C.KD_ALPHAS × T ∈ C.KD_TEMPS per (seed, fold, student). Student memakai imgsz,
epoch & batch yang SAMA dengan baseline Part 3 → selisih mAP murni efek KD.
Baris Excel (results/part4_kd.xlsx): metrik lengkap + baseline_* + delta_* (KD − baseline).

Jalankan: python part4_compression_kd.py [--seed 42] [--fold 0]
"""
from guava_core import (C, DG, PartState, ResultBook, ensure_deps, env_report, fold_name, load_results, log,
                        parse_part_args, part_main, pick_imgsz, promote_final, read_json, run_unit, unit_key,
                        work_dir, write_part_done)

PART = "part4"
METRICS = ["mAP50_box", "mAP50-95_box", "mAP50_mask", "mAP50-95_mask", "precision", "recall"]


def hp_name(a, t):
    return f"a{a}_T{t}"


def grid_for(seed, student, scope):
    full = [(a, t) for a in C.KD_ALPHAS for t in C.KD_TEMPS]
    if scope != "search_then_full" or seed == C.SEEDS[0]:
        return full
    df = load_results("part4_kd")
    if df.empty:
        return full
    df = df[(df["seed"] == C.SEEDS[0]) & (df["model"] == student) & (df["status"] == "OK")]
    if df["fold"].nunique() < len(C.FOLDS) or df["hyperparam"].nunique() < len(full):
        return full  # pencarian di seed pertama belum lengkap
    best = df.groupby("hyperparam")["val_mAP50-95_mask"].mean().idxmax()
    r = df[df["hyperparam"] == best].iloc[0]
    return [(float(r["alpha"]), float(r["T"]))]


def lookup(df, model, seed, fold):
    if df is None or df.empty:
        return None
    r = df[(df["model"] == model) & (df["seed"] == seed) & (df["fold"] == fold)]
    if "technique" in r.columns:
        r = r[r["technique"] == "baseline"]
    return None if r.empty else r.iloc[-1]


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
    students = M.usable(args.model_list or C.KD_STUDENTS, nc, family="edge")
    tokens = sum((M.cache_tokens(s) for s in students), [])
    state = PartState(PART)
    book = ResultBook(C.EXCEL_DIR / "part4_kd.xlsx")

    for seed in args.seeds:
        for fold in args.folds:
            grids = {s: grid_for(seed, s, C.KD_GRID_SCOPE) for s in students}
            keys = [unit_key(s, seed, fold, "kd", hp_name(a, t)) for s in students for a, t in grids[s]]
            if all(state.is_done(k) for k in keys):
                continue
            wd = work_dir(seed, fold)
            tmeta = read_json(wd / "teacher.json")
            if tmeta is None or not (wd / "teacher.pt").exists():
                log.error(f"[KD] {fold_name(seed, fold)}: teacher belum ada (jalankan Part 1/2/3 dulu) → skip")
                continue
            log.info(f"[KD] {fold_name(seed, fold)} teacher = {tmeta['model']} ({tmeta['part']}, "
                     f"{C.TEACHER_SELECT_METRIC}={tmeta['score']:.4f})")
            fd = D.prepare_fold(seed, fold, master)
            teacher = T.KDTeacher(tmeta, wd / "teacher.pt", nc)
            p3 = load_results("part3_edge_baseline")
            trow = None
            for stem in ("part1_yolo", "part2_transformer_cnn", "part3_edge_baseline"):
                trow = lookup(load_results(stem), tmeta["model"], seed, fold)
                if trow is not None:
                    break

            for student in students:
                S = pick_imgsz(seed, fold, student)   # sama dengan baseline Part 3
                brow = lookup(p3, student, seed, fold)
                for a, t in grids[student]:
                    key = unit_key(student, seed, fold, "kd", hp_name(a, t))
                    if state.is_done(key):
                        continue
                    beta = (1 - a) if C.KD_BETA_FROM_ALPHA else C.KD_BETA
                    base = {"part": PART, "model": student, "family": "edge", "seed": seed, "fold": fold,
                            "imgsz": S, "technique": "kd", "hyperparam": hp_name(a, t), "alpha": a, "T": t,
                            "beta": beta, "gamma": C.KD_GAMMA, "teacher_model": tmeta["model"],
                            "teacher_part": tmeta["part"]}
                    if trow is not None:
                        base.update({f"teacher_{sp}_{m}": trow.get(f"{sp}_{m}") for sp in ("val", "test")
                                     for m in ("mAP50-95_mask", "mAP50_mask")})
                    wpath = wd / f"kd_{student}_{hp_name(a, t)}.pt"

                    def fn(attempt, student=student, S=S, a=a, t=t, beta=beta, key=key, base=base, wpath=wpath,
                           brow=brow, seed=seed, fold=fold, teacher=teacher):
                        _, res = T.semantic_unit(
                            book, key, base, attempt, fd, S, C.EPOCHS_EDGE, C.BATCH_EDGE, C.LR["edge"], seed,
                            wpath, make_model=lambda: M.build(student, nc, pretrained=True),
                            make_kd=lambda: T.KDLoss(teacher, a, t, beta, C.KD_GAMMA))
                        if brow is not None:
                            for sp in ("val", "test"):
                                for m in METRICS:
                                    bv = brow.get(f"{sp}_{m}")
                                    res[f"baseline_{sp}_{m}"] = bv
                                    if bv is not None and bv == bv:
                                        res[f"delta_{sp}_{m}"] = res[f"{sp}_{m}"] - bv
                        promote_final(wpath, "kd", f"{student}_{hp_name(a, t)}", seed, fold,
                                      res["val_mAP50-95_mask"], {"imgsz": S, "alpha": a, "T": t})
                        # KD terbaik per (seed, fold, student) → dipakai Part 5
                        bk = f"best_{fold_name(seed, fold)}_{student}"
                        cur = state.get(bk)
                        if cur is None or res["val_mAP50-95_mask"] > cur["score"]:
                            wpath.replace(wd / f"kd_{student}.pt")
                            state.set(bk, {"score": res["val_mAP50-95_mask"], "hp": hp_name(a, t)})
                        wpath.unlink(missing_ok=True)
                        return res

                    run_unit(state, book, key, base, fn, keep_models=tokens + [tmeta["model"]],
                             keep_folds=[fd.name])
                    DG.clean_all(keep_models=tokens + M.cache_tokens(tmeta["model"]), keep_folds=[fd.name],
                                 verbose=False)
            del teacher
            T.torch.cuda.empty_cache() if T.torch.cuda.is_available() else None
            book.flush(include_history=True)

    book.flush(include_history=True)
    if not args.no_final:
        expected = [unit_key(s, sd, f, "kd", hp_name(a, t)) for sd in C.SEEDS for f in C.FOLDS for s in students
                    for a, t in grid_for(sd, s, C.KD_GRID_SCOPE)]
        write_part_done(PART, state, expected, {"students": students})
        DG.clean_all(keep_models=[] if DG.disk_free_gb() < C.CACHE_PURGE_FREE_GB else tokens, keep_folds=[])
        DG.report_disk()


if __name__ == "__main__":
    part_main(PART, main)
