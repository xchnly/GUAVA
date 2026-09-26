"""
part1_dataset_baseline.py — FASE 1: siapkan data + YOLO baseline (kandidat teacher).

1A. Setup: install deps, cek GPU & disk, log → logs/part1.log
1B. Split ulang 80/10/10 × 5-fold stratified × 5 seed, preprocessing baru, cache .npy,
    balancing (400/kelas, max +3000 aug) → results/dataset_split_summary.xlsx
1C. YOLO baseline (yolo11x-seg, yolo26x-seg) per (seed, fold) = 2 × 5 × 5 = 50 run.
    Simpan HANYA best.pt. Eval val+test → results/part1_yolo.xlsx segera tiap run.

Jalankan: python part1_dataset_baseline.py [--seed 42] [--fold 0]
Resume  : jalankan ulang — unit yang sudah selesai di state/part1.json di-skip.
"""
import shutil

from guava_core import (C, DG, PartState, ResultBook, ensure_deps, env_report, log,
                        parse_part_args, part_main, pick_imgsz, promote_final, seed_everything, unit_key,
                        update_teacher, run_unit, write_part_done)

PART = "part1"


def main():
    args = parse_part_args(__doc__)
    ensure_deps()
    env_report()
    DG.report_disk()
    DG.guard_before_training()

    import guava_data as D
    import guava_models as M
    import guava_train as T

    # ---------------- 1B. Data ----------------
    master = D.build_master_cache()
    DG.clean_raw_jpg(C.RAW_DIR)
    nc = master["nc"]
    log.info(f"[DATA] {len(master['items'])} gambar | nc={nc} | {master['names']}")

    models = M.usable(args.model_list or C.YOLO_MODELS, nc, family="yolo")
    keep_tokens = sum((M.cache_tokens(m) for m in models), [])

    state = PartState(PART)
    book = ResultBook(C.EXCEL_DIR / "part1_yolo.xlsx")
    split_book = ResultBook(C.EXCEL_DIR / "dataset_split_summary.xlsx")
    expected = [unit_key(m, s, f) for s in C.SEEDS for f in C.FOLDS for m in models]

    for seed in args.seeds:
        for fold in args.folds:
            if all(state.is_done(unit_key(m, seed, fold)) for m in models) and state.get(f"summary_s{seed}_f{fold}"):
                continue
            seed_everything(seed)
            fd = D.prepare_fold(seed, fold, master)
            if not state.get(f"summary_{fd.name}"):
                split_book.add("summary", D.split_summary(fd))
                state.set(f"summary_{fd.name}", True)

            # ---------------- 1C. YOLO ----------------
            for name in models:
                key = unit_key(name, seed, fold)
                if state.is_done(key):
                    continue
                S = pick_imgsz(seed, fold, name)
                base = {"part": PART, "model": name, "family": "yolo", "seed": seed, "fold": fold, "imgsz": S,
                        "technique": "baseline", "hyperparam": "-"}

                def fn(attempt, name=name, S=S, key=key, base=base, fd=fd, seed=seed, fold=fold):
                    run = f"{name}_{fd.name}"
                    hist = []

                    def h(r):
                        hist.append(r)
                        book.add("history", {"unit": key, "attempt": attempt, "model": name, "seed": seed,
                                             "fold": fold, **r}, flush=False)

                    best, used = T.with_oom_retry(
                        lambda b: T.train_yolo(name, fd, S, C.EPOCHS_YOLO, b, seed, run, h), C.BATCH_YOLO)
                    res = T.eval_yolo(best, fd, S)
                    col = next((c for c in (hist[0] if hist else {}) if "mAP50-95(M)" in c), None)
                    res["epochs_run"] = len(hist)
                    res["best_epoch"] = (max(range(len(hist)), key=lambda i: hist[i][col]) + 1) if col else None
                    res["batch"] = used
                    promote_final(best, "yolo", name, seed, fold, res["val_mAP50-95_mask"], {"imgsz": S})
                    update_teacher(seed, fold, name, "yolo", "yolo", res[C.TEACHER_SELECT_METRIC], best, S, PART)
                    shutil.rmtree(C.RUNS_DIR / run, ignore_errors=True)
                    return res

                run_unit(state, book, key, base, fn, keep_models=keep_tokens, keep_folds=[fd.name])
                DG.clean_all(keep_models=keep_tokens, keep_folds=[fd.name], verbose=False)
            DG.clean_datasets(C.DATASETS, keep_folds=[])   # dataset YOLO sementara fold ini
            book.flush(include_history=True)

    book.flush(include_history=True)
    split_book.flush()
    if not args.no_final:
        write_part_done(PART, state, expected, {"models": models})
        free = DG.disk_free_gb()
        DG.clean_all(keep_models=[] if free < C.CACHE_PURGE_FREE_GB else keep_tokens, keep_folds=[])
        DG.report_disk()


if __name__ == "__main__":
    part_main(PART, main)
