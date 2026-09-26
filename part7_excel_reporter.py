"""
part7_excel_reporter.py — FASE 7: gabung semua hasil → MASTER_RESULTS.xlsx + plot ringkas.

Sheet: dataset_split_summary, part1_yolo, part2_transformer_cnn, part3_edge_baseline, part4_kd,
       part5_pruning, part5_quantization, part6_champion, kd_significance, all_runs_flat,
       history_* (per-epoch).
Ditulis dengan xlsxwriter mode constant_memory (write-only, hemat RAM).
Plot (results/summary_plots/*.png): perbandingan family, trade-off edge (mAP | size | latency),
mAP vs latency, gain KD per student.
Bisa dijalankan kapan saja (juga di tengah eksperimen) — membaca hasil yang sudah ada.
"""
import math

from guava_core import (C, PartState, ResultBook, env_report, load_results, log, parse_part_args, part_main,
                        write_part_done)

PART = "part7"
SOURCES = [
    ("dataset_split_summary", "dataset_split_summary", "summary"),
    ("part1_yolo", "part1_yolo", "results"),
    ("part2_transformer_cnn", "part2_transformer_cnn", "results"),
    ("part3_edge_baseline", "part3_edge_baseline", "results"),
    ("part4_kd", "part4_kd", "results"),
    ("part5_pruning", "part5_pruning", "results"),
    ("part5_quantization", "part5_quantization", "results"),
    ("part6_champion", "part6_champion", "candidates"),
]
FLAT_COLS = ["part", "model", "family", "fold", "seed", "imgsz", "teknik", "hyperparam", "mAP50_box", "mAP50-95_box",
             "mAP50_mask", "mAP50-95_mask", "precision", "recall", "val_mAP50-95_mask", "params_M", "flops_G",
             "size_MB", "latency_ms", "latency_gpu_ms", "status", "duration_min", "vram_peak_GB", "timestamp"]
# Palet kategorikal tetap (urutan tidak di-cycle) — warna mengikuti entitas, bukan peringkat
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TECH_ORDER = ["baseline", "kd", "prune", "ptq_int8_ort", "ptq_int8_ov", "qat_int8"]
FAMILY_ORDER = ["yolo", "transformer", "cnn", "edge"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def flat(frames):
    import pandas as pd
    out = []
    for name, df in frames.items():
        if df.empty or not name.startswith("part") or name == "part6_champion":
            continue
        d = pd.DataFrame()
        d["part"] = df.get("part", name.split("_")[0])
        for c in ("model", "family", "fold", "seed", "imgsz", "hyperparam", "params_M", "flops_G", "size_MB",
                  "latency_ms", "latency_gpu_ms", "status", "duration_min", "vram_peak_GB", "timestamp",
                  "val_mAP50-95_mask"):
            d[c] = df.get(c)
        d["teknik"] = df.get("technique")
        for m in ("mAP50_box", "mAP50-95_box", "mAP50_mask", "mAP50-95_mask", "precision", "recall"):
            d[m] = df.get(f"test_{m}")
        out.append(d[FLAT_COLS])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=FLAT_COLS)


def kd_significance(kd, p3):
    """Uji berpasangan (seed, fold): KD (alpha,T terbaik by val) vs baseline student yang sama."""
    import pandas as pd
    from scipy import stats
    rows = []
    if kd.empty or p3.empty:
        return pd.DataFrame(rows)
    kd = kd[kd["status"] == "OK"]
    p3 = p3[p3["status"] == "OK"]
    for s, g in kd.groupby("model"):
        best_hp = g.groupby("hyperparam")["val_mAP50-95_mask"].mean().idxmax()
        k = g[g["hyperparam"] == best_hp][["seed", "fold", "test_mAP50-95_mask", "test_mAP50_mask"]]
        b = p3[p3["model"] == s][["seed", "fold", "test_mAP50-95_mask", "test_mAP50_mask"]]
        m = k.merge(b, on=["seed", "fold"], suffixes=("_kd", "_base"))
        for met in ("test_mAP50-95_mask", "test_mAP50_mask"):
            d = m[f"{met}_kd"] - m[f"{met}_base"]
            r = {"student": s, "best_hp": best_hp, "metric": met, "n_pairs": len(d),
                 "baseline_mean": m[f"{met}_base"].mean(), "kd_mean": m[f"{met}_kd"].mean(),
                 "delta_mean": d.mean(), "delta_std": d.std(), "kd_wins": int((d > 0).sum())}
            if len(d) >= 3 and d.std() > 0:
                r["ttest_p"] = stats.ttest_rel(m[f"{met}_kd"], m[f"{met}_base"]).pvalue
                try:
                    r["wilcoxon_p"] = stats.wilcoxon(d).pvalue
                except ValueError:
                    r["wilcoxon_p"] = math.nan
                r["cohen_dz"] = d.mean() / d.std()
            rows.append(r)
    return pd.DataFrame(rows)


def write_master(frames, extra):
    import xlsxwriter
    tmp = C.MASTER_XLSX.with_suffix(".tmp.xlsx")
    wb = xlsxwriter.Workbook(str(tmp), {"constant_memory": True, "nan_inf_to_errors": True})
    bold = wb.add_format({"bold": True, "bg_color": "#f0efec"})
    for name, df in list(frames.items()) + list(extra.items()):
        ws = wb.add_worksheet(name[:31])
        if df is None or df.empty:
            ws.write_row(0, 0, ["(belum ada data)"])
            continue
        ws.write_row(0, 0, [str(c) for c in df.columns], bold)
        ws.freeze_panes(1, 0)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            ws.write_row(i, 0, [None if (isinstance(v, float) and math.isnan(v)) else
                                (v if isinstance(v, (int, float, str, type(None))) else str(v)) for v in row])
    wb.close()
    tmp.replace(C.MASTER_XLSX)
    log.info(f"[REPORT] {C.MASTER_XLSX} ({C.MASTER_XLSX.stat().st_size / 1e6:.1f} MB)")


# =============================================================================
# PLOT
# =============================================================================
def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _colors(keys, order):
    known = [k for k in order if k in keys] + sorted(k for k in keys if k not in order)
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(known)}


def plots(frames, champ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Patch
    out = C.EXCEL_DIR / "summary_plots"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE})

    # 1. Perbandingan family (Part 1/2/3) — mean ± std test mAP50-95 mask
    base = pd.concat([frames[k] for k in ("part1_yolo", "part2_transformer_cnn", "part3_edge_baseline")
                      if not frames[k].empty], ignore_index=True) if any(
        not frames[k].empty for k in ("part1_yolo", "part2_transformer_cnn", "part3_edge_baseline")) else None
    if base is not None:
        base = base[base["status"] == "OK"]
        g = base.groupby(["family", "model"])["test_mAP50-95_mask"].agg(["mean", "std"]).reset_index()
        g["fo"] = g["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
        g = g.sort_values(["fo", "mean"])
        cmap = _colors(set(g["family"]), FAMILY_ORDER)
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(g) + 1.2))
        ax.barh(g["model"] + " (" + g["family"] + ")", g["mean"], xerr=g["std"].fillna(0), height=0.6,
                color=[cmap[f] for f in g["family"]], edgecolor=SURFACE, linewidth=2,
                error_kw={"ecolor": INK2, "lw": 1})
        _style(ax)
        ax.set_xlabel("test mAP50-95 (mask), mean ± std over seed × fold")
        ax.legend(handles=[Patch(color=cmap[f], label=f) for f in cmap], frameon=False, loc="lower right")
        ax.set_title("Family comparison", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "family_comparison.png", dpi=200)
        plt.close(fig)

    # 2. Trade-off edge: tiga panel terpisah (tanpa dual axis) dengan urutan baris sama
    if champ is not None and not champ.empty:
        top = champ.sort_values("score", ascending=False).head(15).iloc[::-1]
        labels = top["model"] + " · " + top["technique"] + " · " + top["hyperparam"].astype(str)
        cmap = _colors(set(top["technique"]), TECH_ORDER)
        cols = [cmap[t] for t in top["technique"]]
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.34 * len(top) + 1.4), sharey=True)
        for ax, col, xl in zip(axes, ("test_mAP50_95_mask", "size_MB", "latency_cpu_ms"),
                               ("test mAP50-95 mask ↑", "size (MB) ↓", "CPU latency, batch 1 (ms) ↓")):
            ax.barh(labels, top[col], height=0.6, color=cols, edgecolor=SURFACE, linewidth=2)
            _style(ax)
            ax.set_xlabel(xl)
        axes[0].legend(handles=[Patch(color=cmap[t], label=t) for t in cmap], frameon=False, loc="lower right",
                       fontsize=7)
        fig.suptitle("Edge candidates: accuracy vs size vs latency (top 15 by champion score)", x=0.01,
                     ha="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "edge_tradeoff.png", dpi=200)
        plt.close(fig)

        # 3. mAP vs latency (scatter, satu sumbu y)
        cmap = _colors(set(champ["technique"]), TECH_ORDER)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t, g in champ.groupby("technique"):
            ax.scatter(g["latency_cpu_ms"], g["test_mAP50_95_mask"], s=18 + 6 * g["size_MB"].clip(upper=30),
                       color=cmap[t], edgecolor=SURFACE, linewidth=1.5, label=t, alpha=0.9)
        best = champ.iloc[0]
        ax.annotate(f"champion: {best['model']} ({best['technique']})",
                    (best["latency_cpu_ms"], best["test_mAP50_95_mask"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=8, color=INK)
        ax.set_xscale("log")
        _style(ax, "both")
        ax.set_xlabel("CPU latency (ms, log) — marker size ∝ model size")
        ax.set_ylabel("test mAP50-95 mask")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Accuracy vs latency", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "map_vs_latency.png", dpi=200)
        plt.close(fig)

    # 4. Gain KD per student
    sig = frames.get("_kd_sig")
    if sig is not None and not sig.empty:
        s = sig[sig["metric"] == "test_mAP50-95_mask"].sort_values("delta_mean")
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.2))
        y = range(len(s))
        ax.barh([i + 0.2 for i in y], s["baseline_mean"], height=0.38, color=PALETTE[0], label="baseline",
                edgecolor=SURFACE, linewidth=2)
        ax.barh([i - 0.2 for i in y], s["kd_mean"], height=0.38, color=PALETTE[1], label="KD (best α,T)",
                edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{r.student}  Δ={r.delta_mean:+.3f}" +
                            (f", p={r.ttest_p:.3f}" if "ttest_p" in s and r.ttest_p == r.ttest_p else "")
                            for r in s.itertuples()])
        _style(ax)
        ax.set_xlabel("test mAP50-95 mask (mean over seed × fold)")
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("Knowledge distillation gain per student", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "kd_gain.png", dpi=200)
        plt.close(fig)
    log.info(f"[REPORT] plot → {out}")


def main():
    args = parse_part_args(__doc__)
    env_report()
    frames = {name: load_results(stem, sheet) for name, stem, sheet in SOURCES}
    sig = kd_significance(frames["part4_kd"], frames["part3_edge_baseline"])
    extra = {"kd_significance": sig, "all_runs_flat": flat(frames)}
    for name, stem, _ in SOURCES:
        h = load_results(stem, "history")
        if not h.empty:
            extra[f"history_{name.replace('part', 'p')}"[:31]] = h
    write_master(frames, extra)
    frames["_kd_sig"] = sig
    try:
        plots(frames, frames["part6_champion"])
    except Exception as e:
        log.warning(f"[REPORT] plot gagal: {e}")
    for b in (ResultBook(C.EXCEL_DIR / f"{stem}.xlsx") for _, stem, _ in SOURCES):
        b.flush(include_history=True)   # sinkronkan part*.xlsx dengan journal
    state = PartState(PART)
    state.mark_done("report")
    if not args.no_final:
        write_part_done(PART, state, ["report"])


if __name__ == "__main__":
    part_main(PART, main)
