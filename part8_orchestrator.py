"""
part8_orchestrator.py — FASE 8: otak utama. Jalankan semua part dengan checkpoint resume.

Strategi hemat disk: loop (seed, fold) DI LUAR, part DI DALAM → hanya satu set bobot kerja
(teacher + student fold aktif) di disk pada satu waktu. Tiap part dijalankan sebagai subprocess
(`python partN.py --seed s --fold f --no-final`) → crash/OOM satu part tidak mematikan
orchestrator, dan memori GPU bersih tiap part.

Prioritas (C.ORCH_ORDER):  P1 part1 → part3 → part4   |   P2 part2 → part5   |   P3 part6 → part7
Disk < C.DISK_TIER2_PURGE_GB → part2 & part5 ditunda ke akhir. Disk < 0.5 GB → stop, state tersimpan.

Jalankan: python part8_orchestrator.py            (resume otomatis — cukup jalankan ulang)
          python part8_orchestrator.py --only part1,part3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from guava_core import C, DG, EXIT_DISK, fold_name, is_part_done, log, read_json, setup_logging, write_json

HERE = Path(__file__).resolve().parent
ORCH_STATE = C.STATE_DIR / "orchestrator.json"
DEFERRABLE = {"part2", "part5"}


def run_part(part, extra_args, st, tag):
    script = HERE / C.PART_SCRIPTS[part]
    for attempt in range(1, C.MAX_RETRY + 2):
        DG.guard_before_training()
        t0 = time.time()
        log.info(f"▶ {part} {tag} (attempt {attempt})")
        rc = subprocess.call([sys.executable, str(script), *extra_args], cwd=str(HERE))
        dt = (time.time() - t0) / 60
        if rc == 0:
            log.info(f"✔ {part} {tag} selesai ({dt:.1f} min)")
            st["done"].append(f"{part}|{tag}")
            write_json(ORCH_STATE, st)
            return True
        if rc == EXIT_DISK:
            log.critical(f"✘ {part} {tag}: DISK HARD STOP — orchestrator berhenti. Bebaskan disk lalu jalankan ulang.")
            write_json(ORCH_STATE, st)
            sys.exit(EXIT_DISK)
        log.error(f"✘ {part} {tag} exit={rc} ({dt:.1f} min)")
        DG.clean_all(verbose=False, purge_model_cache=False)
    st["failed"].append(f"{part}|{tag}")
    write_json(ORCH_STATE, st)
    log.error(f"{part} {tag} gagal {C.MAX_RETRY + 1}x → lanjut (unit yang gagal tercatat di Excel/state)")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="subset part, mis. part1,part3,part4")
    args = ap.parse_args()
    setup_logging("run_master")
    only = [p for p in args.only.split(",") if p]
    order = [p for p in C.ORCH_ORDER if not only or p in only]
    finals = [p for p in C.ORCH_FINAL if not only or p in only]
    st = read_json(ORCH_STATE, {"done": [], "failed": [], "deferred": []})
    log.info(f"=== ORCHESTRATOR | profile={C.PROFILE} | seeds={C.SEEDS} | folds={C.FOLDS} | order={order} ===")
    DG.report_disk()

    for seed in C.SEEDS:
        for fold in C.FOLDS:
            tag = fold_name(seed, fold)
            for part in order:
                if is_part_done(part) or f"{part}|{tag}" in st["done"]:
                    continue
                if part in DEFERRABLE and DG.disk_free_gb() < C.DISK_TIER2_PURGE_GB:
                    if f"{part}|{tag}" not in st["deferred"]:
                        st["deferred"].append(f"{part}|{tag}")
                        write_json(ORCH_STATE, st)
                    log.warning(f"[DISK] {part} {tag} ditunda (free < {C.DISK_TIER2_PURGE_GB} GB)")
                    continue
                run_part(part, ["--seed", str(seed), "--fold", str(fold), "--no-final"], st, tag)
                DG.clean_all(keep_folds=[tag], verbose=False, purge_model_cache=False)
            # bobot kerja fold ini sudah tidak dibutuhkan bila part5 tidak ditunda
            if f"part5|{tag}" not in st["deferred"] and "part5" in order:
                from guava_core import finalize_fold
                finalize_fold(seed, fold)
            DG.clean_all(keep_folds=[], verbose=False,
                         purge_model_cache=DG.disk_free_gb() < C.CACHE_PURGE_FREE_GB)
            # laporan sementara tiap fold (Excel master selalu terbaru)
            if "part7" in finals:
                subprocess.call([sys.executable, str(HERE / C.PART_SCRIPTS["part7"]), "--no-final"], cwd=str(HERE))

    # Part yang ditunda (disk sempat kritis)
    for item in list(st["deferred"]):
        part, tag = item.split("|")
        if f"{part}|{tag}" in st["done"] or is_part_done(part):
            st["deferred"].remove(item)
            continue
        s, f = tag[1:].split("_f")
        if DG.disk_free_gb() >= C.DISK_MIN_FREE_GB and run_part(part, ["--seed", s, "--fold", f, "--no-final"], st, tag):
            st["deferred"].remove(item)
            write_json(ORCH_STATE, st)

    # Finalisasi tiap part (tulis partN_done.json; unit yang sudah selesai di-skip → cepat)
    for part in order:
        if not is_part_done(part):
            run_part(part, [], st, "final")
    for part in finals:
        run_part(part, [], st, "final")
    DG.clean_all(keep_folds=[], purge_model_cache=True)
    DG.report_disk(top_n=15)
    log.info(f"=== SELESAI | gagal: {st['failed'] or '-'} | ditunda tersisa: {st['deferred'] or '-'} ===")


if __name__ == "__main__":
    try:
        main()
    except DG.DiskHardStop as e:
        log.critical(f"DISK HARD STOP: {e}")
        sys.exit(EXIT_DISK)
