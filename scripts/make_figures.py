"""
Figures for the RIFT v2 re-analysis. Every plotted value is read from ./outputs (written by rift_pipeline.py);
nothing is typed in by hand. 600 dpi PNG at MDPI full text width (17.1 cm).
Palette: reference categorical slots 1-3 (validated all-pairs; aqua is < 3:1 on white, so every figure carries a
legend and/or direct labels). Status colours (good/critical) only for the Good/Poor classification, always labelled.
Error measure shown in figures: WAPE (sum|e| / sum|y|), because plain MAPE is dominated by depths near 0.01 m.
"""
from pathlib import Path
import json
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import geopandas as gpd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
SHAC = os.environ.get("RIFT_SHAC_SHP", str(HERE.parent / "data" / "raw" / "INV_ACUIFEROS_SHAC.shp"))  # DGA SHAC layer
W, DPI = 6.73, 600
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MODELS = ["Decision Tree", "Random Forest", "Extra Trees"]
MC = {"Decision Tree": "#2a78d6", "Random Forest": "#eb6834", "Extra Trees": "#1baf7a"}
BASE_C = "#b8b6ae"; GOOD, POOR = "#0ca30c", "#d03b3b"
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DESIGNS = ["random", "well", "spatial", "temporal_per_well", "temporal_global"]
NAME_FIX = {"PUEBLO QULIMARI": "Pueblo Quilimarí"}   # source spelling of the locality Quilimarí
DLAB = {"random": "Random", "well": "Well-based", "spatial": "Spatial block",
        "temporal_per_well": "Chronological\n(per well)", "temporal_global": "Chronological\n(global origin)"}
TARGET, WELL = "Depth to water (m)", "well_id"
METS = [("R2", "Testing R²"), ("RMSE", "Testing RMSE (m)"), ("WAPE", "Testing WAPE (%)")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": DPI})

def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)

def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")

def shac():
    try: return gpd.read_file(SHAC).to_crs(4326)
    except Exception: return None

def basemap(ax, g):
    if g is not None: g.plot(ax=ax, color="#f0efec", edgecolor="#d6d4cc", linewidth=0.2)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([-73, -70]); ax.set_xticklabels(["73° W", "70° W"])
    ax.set_yticks([-40, -35, -30, -25, -20]); ax.set_yticklabels(["40° S", "35° S", "30° S", "25° S", "20° S"])

def mfmt(v, f):   # typographic minus in figure text
    return format(v, f).replace("-", "−")

def preds(design, seed=42, tag="main"):
    return pd.read_parquet(OUT / "predictions" / f"{tag}_{design}_seed{seed}.parquet")

def well_table(t):
    return t.groupby(WELL).agg(lon=("well_lon", "first"), lat=("well_lat", "first"), name=("Name", "first"),
                               dtw=(TARGET, "mean"), n=(TARGET, "size"))

def fig02(t, g):
    wt = well_table(t)
    sp = pd.read_csv(OUT / "splits_main.csv")
    fig, axes = plt.subplots(1, 5, figsize=(W, 4.9), sharey=True)
    cols = {"Training only": MC["Decision Tree"], "Testing only": MC["Random Forest"], "Both": MC["Extra Trees"],
            "Excluded": BASE_C}
    for k, (ax, d) in enumerate(zip(axes, DESIGNS)):
        basemap(ax, g); p = preds(d)
        is_te = np.isin(np.arange(len(t)), p.row.to_numpy())
        if d == "temporal_global":   # training = before the origin; later records at untrained wells are excluded
            is_tr = (t.month < pd.Timestamp(sp[sp.design == d].origin.iloc[0])).to_numpy()
        else:
            is_tr = ~is_te
        te_w = pd.Series(is_te, index=t.index).groupby(t[WELL]).any()
        tr_w = pd.Series(is_tr, index=t.index).groupby(t[WELL]).any()
        cls = pd.Series(np.select([tr_w & te_w, tr_w, te_w], ["Both", "Training only", "Testing only"], "Excluded"), index=tr_w.index)
        for c, col in cols.items():
            s_ = wt.loc[cls[cls == c].index]
            if len(s_): ax.scatter(s_.lon, s_.lat, s=3, color=col, lw=0, label=f"{c} ({len(s_)})")
        if d == "spatial":
            import rift_pipeline as rp
            _, lon_e, lat_e = rp.grid_cells(t)
            for x in lon_e: ax.axvline(x, color=MUTED, lw=0.3)
            for y in lat_e: ax.axhline(y, color=MUTED, lw=0.3)
        ax.set_title(DLAB[d], fontsize=7.2); ax.set_xlim(-74, -67.5); ax.set_ylim(-42.5, -17.5)
        if k: ax.set_ylabel("")
        ax.set_xlabel("")
        panel(ax, "abcde"[k]); ax.legend(loc="lower left", fontsize=5.2, frameon=False, markerscale=2, handletextpad=0.1,
                                         borderaxespad=0.1, labelspacing=0.3)
    save(fig, "Figure2_split_maps.png")

def fig03(t, g):
    w = well_table(t)
    fig = plt.figure(figsize=(W, 4.2))
    ax = fig.add_axes([0.02, 0.08, 0.36, 0.86]); basemap(ax, g)
    sc = ax.scatter(w.lon, w.lat, c=w.dtw, s=2 + 18 * w.n / w.n.max(), cmap=BLUE,
                    norm=LogNorm(max(w.dtw.min(), .1), w.dtw.max()), lw=0.2, edgecolor="white")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Mean depth to water (m, log scale)")
    ax.set_xlim(-74, -67.5); ax.set_ylim(-42.5, -17.5); panel(ax, "a")
    for nn in (50, 200, 400):                                   # marker-size key: well-months per well
        ax.scatter([], [], s=2 + 18 * nn / w.n.max(), color=INK2, lw=0, label=f"{nn}")
    ax.legend(title="Well-months", loc="lower left", fontsize=5.5, title_fontsize=5.8, frameon=False, handletextpad=0.2,
              labelspacing=0.3, borderaxespad=0.2)
    ax2 = fig.add_axes([0.55, 0.58, 0.42, 0.36]); ax3 = fig.add_axes([0.55, 0.10, 0.42, 0.36]); y = t[TARGET]
    ax2.hist(y, bins=80, color=MC["Decision Tree"], edgecolor="white", linewidth=0.2)
    ax2.set_xlabel("Depth to water (m)"); ax2.set_ylabel("Well-months"); panel(ax2, "b")
    yc = t.month.dt.year.value_counts().sort_index()
    ax3.bar(yc.index, yc.values, width=0.8, color=MC["Decision Tree"], edgecolor="white", linewidth=0.2)
    ax3.set_xlabel("Year"); ax3.set_ylabel("Well-months"); panel(ax3, "c")
    save(fig, "Figure3_data_distribution.png")

def best_baseline(m):
    b = m[~m.model.isin(MODELS)]
    return b.groupby(["design", "model"])["RMSE"].mean().reset_index().sort_values("RMSE").groupby("design").first()["model"]

def fig04(m):
    """(a) R2, (b) RMSE for learners + best admissible baseline; (c) skill score 1 - MSE/MSE_ref, paired by seed."""
    bb = best_baseline(m)
    fig, axes = plt.subplots(3, 1, figsize=(W, 6.8), sharex=True)
    x = np.arange(len(DESIGNS)); wbar = 0.19; series = MODELS + ["Reference baseline"]
    R2_FLOOR = -1.0
    for ax, met, lab in [(axes[0], "R2", "Testing R²"), (axes[1], "RMSE", "Testing RMSE (m)")]:
        for j, s_ in enumerate(series):
            vals, errs = [], []
            for d in DESIGNS:
                mm = m[(m.design == d) & (m.model == (bb.get(d) if s_ == "Reference baseline" else s_))][met]
                vals.append(mm.mean()); errs.append(mm.std() if len(mm) > 1 else 0)
            xs = x + (j - 1.5) * (wbar + 0.02); v = np.array(vals)
            shown = np.maximum(v, R2_FLOOR) if met == "R2" else v
            ax.bar(xs, shown, width=wbar, color=MC.get(s_, BASE_C), label=s_, yerr=np.where(shown == v, errs, 0),
                   error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK2})
            if met == "R2":
                for xi, vi in zip(xs, v):
                    if vi < R2_FLOOR: ax.text(xi, R2_FLOOR + 0.03, f"{vi:.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=5.5, color=INK, rotation=90)
        ax.set_ylabel(lab); ax.axhline(0, color=INK2, lw=0.6)
    axes[0].set_ylim(R2_FLOOR, 1.0)
    ax = axes[2]
    for j, mdl in enumerate(MODELS):
        vals, errs = [], []
        for d in DESIGNS:
            a = m[(m.design == d) & (m.model == mdl)].set_index("seed").RMSE
            b = m[(m.design == d) & (m.model == bb[d])].set_index("seed").RMSE.reindex(a.index)
            ss = 1 - (a / b) ** 2; vals.append(ss.mean()); errs.append(ss.std() if len(ss) > 1 else 0)
        v = np.array(vals); shown = np.maximum(v, -1.0)
        xs = x + (j - 1) * (wbar + 0.02)
        ax.bar(xs, shown, width=wbar, color=MC[mdl], yerr=np.where(shown == v, errs, 0), error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK2})
        for xi, vi in zip(xs, v):
            if vi < -1.0: ax.text(xi, -0.97, f"{vi:.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=5.5, color=INK, rotation=90)
    ax.axhline(0, color=INK, lw=0.8); ax.set_ylim(-1.0, 0.4); ax.set_ylabel("Skill vs. reference baseline")
    ax.text(len(DESIGNS) - 0.5, 0.37, "Above zero: learner beats the reference baseline", fontsize=6.3, color=INK2, ha="right", va="top")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels([DLAB[d] for d in DESIGNS])
    for k, ax in enumerate(axes): panel(ax, "abc"[k])
    save(fig, "Figure4_design_comparison.png")
    bb.to_frame("best_baseline").to_csv(FIG / "Figure4_best_baselines.csv")

def fig05(m):
    fig, axes = plt.subplots(2, 3, figsize=(W, 5.1)); axes = axes.ravel(); lim = (0.01, 150)
    for k, d in enumerate(DESIGNS):
        ax = axes[k]; p = preds(d)
        hb = ax.hexbin(p.y_true.clip(lower=0.05), p["Extra Trees"].clip(lower=0.05), gridsize=45, xscale="log",
                       yscale="log", cmap=BLUE, bins="log", mincnt=1, linewidths=0)
        ax.plot(lim, lim, color=INK2, lw=0.7, ls="--")
        r = m[(m.design == d) & (m.seed == 42) & (m.model == "Extra Trees")].iloc[0]
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"$\\bf{{({'abcde'[k]})}}$  {DLAB[d].replace(chr(10), ' ')}\nR² = {mfmt(r.R2, '.2f')}, RMSE = {r.RMSE:.1f} m\n"
                     f"n = {int(r.n_test):,} test well-months", fontsize=6.8, loc="left")
        ax.set_xlabel("Observed depth (m)"); ax.set_ylabel("Predicted depth (m)")
    cb = fig.colorbar(hb, ax=axes[5], fraction=0.9, aspect=12); cb.set_label("Well-months per cell (log)"); axes[5].axis("off")
    fig.tight_layout(); save(fig, "Figure5_predicted_vs_observed.png")

BL = {"Well mean (training records)": ("Well training mean", "#52514e", "--", "s"),
      "Last observation (persistence)": ("Last observation", "#0b0b0b", ":", "D")}

def fig06():
    """RMSE by horizon (a, b) and in the three sensitivity analyses (c-e), learners against admissible baselines."""
    hz = pd.read_csv(OUT / "horizon_errors.csv"); sb = pd.read_csv(OUT / "sensitivity_baselines.csv")
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.9)); axes = axes.ravel()
    for k, d in enumerate(["temporal_per_well", "temporal_global"]):
        ax = axes[k]; q = hz[hz.design == d]; bins = list(dict.fromkeys(q.horizon_bin)); x = np.arange(len(bins))
        for mdl in MODELS:
            v = q[q.model == mdl].set_index("horizon_bin").reindex(bins).RMSE
            ax.plot(x, v, color=MC[mdl], lw=1.4, marker="o", ms=3.2, label=mdl)
        for b, (lab, col, ls, mk) in BL.items():
            v = q[q.model == b].set_index("horizon_bin").reindex(bins).RMSE
            ax.plot(x, v, color=col, lw=1.2, ls=ls, marker=mk, ms=3, label=lab)
        labs = [f"{int(a)}–{int(b_)}" if b_ < 50 else f"≥{int(a)}" for a, b_ in q.drop_duplicates("horizon_bin")[["h_lo", "h_hi"]].values]
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("Horizon (years)"); ax.set_ylabel("Testing RMSE (m)")
        ax.set_title(DLAB[d].replace(chr(10), " "), fontsize=7.5); panel(ax, "ab"[k])
    specs = [(pd.read_csv(OUT / "sensitivity_min_records.csv"), "min_records", "min_records", "Minimum well-months per well", "Chronological (per well)"),
             (pd.read_csv(OUT / "sensitivity_ratio_per_well.csv"), "train_fraction", "train_fraction", "Training fraction per well", "Chronological (per well)"),
             (pd.read_csv(OUT / "sensitivity_origin_global.csv"), "origin_percentile", "origin_percentile", "Origin (percentile of record months)", "Chronological (global origin)")]
    for k, (df, xcol, an, xl, title) in enumerate(specs):
        ax = axes[2 + k]
        for mdl in MODELS:
            s_ = df[df.model == mdl].sort_values(xcol); ax.plot(s_[xcol], s_.RMSE, color=MC[mdl], lw=1.4, marker="o", ms=3.2)
        for b, (lab, col, ls, mk) in BL.items():
            s_ = sb[(sb.analysis == an) & (sb.model == b)].sort_values("value"); ax.plot(s_.value, s_.RMSE, color=col, lw=1.2, ls=ls, marker=mk, ms=3)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel("Testing RMSE (m)"); ax.set_title(title, fontsize=7.5); panel(ax, "cde"[k])
        if xcol == "min_records":
            ax.set_xscale("log"); ax.set_xticks([3, 9, 27, 54, 81]); ax.set_xticklabels([3, 9, 27, 54, 81]); ax.minorticks_off()
    h, l = axes[0].get_legend_handles_labels(); axes[5].axis("off")
    axes[5].legend(h, l, loc="center", frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "Figure6_horizon_sensitivity.png")

def chrono_with_baselines(t, design="temporal_per_well"):
    """Seed-42 learner predictions plus the admissible baselines, per test well-month (recomputed, deterministic)."""
    import rift_pipeline as rp
    tr, te, _ = rp.DESIGNS[design](t, None)
    p = preds(design); assert np.array_equal(p.row.to_numpy(), te)
    for b, v in rp.baseline_predictions(t, tr, te, design).items(): p[b] = v
    return p

def fig07(t, g):
    p = chrono_with_baselines(t); wt = well_table(t)
    fig = plt.figure(figsize=(W, 3.9))
    ax = fig.add_axes([0.07, 0.13, 0.27, 0.74])
    err = (p["Extra Trees"] - p.y_true).abs(); good = err <= 5
    ax.scatter(p.y_true[~good], p["Extra Trees"][~good], s=0.5, color=POOR, lw=0, rasterized=True, label=f"Poor, error > 5 m ({100*(~good).mean():.1f}%)")
    ax.scatter(p.y_true[good], p["Extra Trees"][good], s=0.5, color=GOOD, lw=0, rasterized=True, label=f"Good, error ≤ 5 m ({100*good.mean():.1f}%)")
    mx = max(p.y_true.max(), p["Extra Trees"].max()); ax.plot([0, mx], [0, mx], color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("Observed depth (m)"); ax.set_ylabel("Predicted depth, Extra Trees (m)")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", markerscale=7, handletextpad=0.1); panel(ax, "a")
    se = p.assign(e_et=(p["Extra Trees"] - p.y_true) ** 2, e_p=(p["Last observation (persistence)"] - p.y_true) ** 2)
    pw = se.groupby("well").agg(rmse=("e_et", "mean"), mse_p=("e_p", "mean"), n_test=("e_et", "size"))
    pw["skill"] = 1 - pw.rmse / pw.mse_p; pw["rmse"] = pw.rmse ** 0.5
    loc = wt.join(pw, how="inner")
    ax = fig.add_axes([0.40, 0.06, 0.26, 0.86]); basemap(ax, g)
    sc = ax.scatter(loc.lon, loc.lat, c=loc.rmse, cmap=BLUE, s=5, lw=0.2, edgecolor="white", norm=LogNorm(max(loc.rmse.min(), .05), loc.rmse.max()))
    cb = fig.colorbar(sc, ax=ax, fraction=0.07, pad=0.02); cb.set_label("Per-well test RMSE, Extra Trees (m)", fontsize=7)
    ax.set_xlim(-74, -67.5); ax.set_ylim(-42.5, -17.5); panel(ax, "b")
    from matplotlib.colors import TwoSlopeNorm
    DIV = LinearSegmentedColormap.from_list("div", ["#b8541f", "#eb6834", "#f6c3a6", "#e1e0d9", "#9fc3ef", "#3987e5", "#104281"])
    ax = fig.add_axes([0.71, 0.06, 0.26, 0.86]); basemap(ax, g); ax.set_yticklabels([])
    order = loc.sort_values("skill", key=lambda s: -s.abs())[::-1]
    sc = ax.scatter(order.lon, order.lat, c=order.skill.clip(-3, 1), cmap=DIV, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1), s=5, lw=0.2, edgecolor="white")
    cb = fig.colorbar(sc, ax=ax, fraction=0.07, pad=0.02, extend="min"); cb.set_label("Per-well skill vs. last observation", fontsize=7)
    ax.set_xlim(-74, -67.5); ax.set_ylim(-42.5, -17.5); panel(ax, "c")
    save(fig, "Figure7_chronological_diagnostics.png")
    json.dump({"wells": int(len(pw)), "wells_skill_positive": int((pw.skill > 0).sum()),
               "good_pct": {m_: float(100 * ((p[m_] - p.y_true).abs() <= 5).mean()) for m_ in MODELS + ["Last observation (persistence)", "Well mean (training records)"]}},
              open(FIG / "Figure7_summary.json", "w"), indent=1)

def fig08(t):
    """Six wells chosen across the distribution of per-well skill of Extra Trees against the last observation."""
    p = chrono_with_baselines(t); wt = well_table(t)
    n_te = p.groupby("well").size()
    rm = p.assign(se=(p["Extra Trees"] - p.y_true) ** 2).groupby("well").se.mean().pow(0.5)
    rp_ = p.assign(se=(p["Last observation (persistence)"] - p.y_true) ** 2).groupby("well").se.mean().pow(0.5)
    elig = rm[(wt.n.reindex(rm.index) >= 60) & (n_te.reindex(rm.index) >= 15)].index
    skill = (1 - (rm / rp_) ** 2).loc[elig].sort_values()
    qs = [0.10, 0.30, 0.50, 0.70, 0.85, 0.95]; picks = [skill.index[int(q * (len(skill) - 1))] for q in qs]
    test_rows = set(p.row)
    fig, axes = plt.subplots(3, 2, figsize=(W, 6.9))
    for k, (ax, wid) in enumerate(zip(axes.ravel(), picks)):
        s_ = t[t[WELL] == wid].sort_values("month"); q = p[p.well == wid].sort_values("month")
        last_train = s_[~s_.index.isin(test_rows)].month.max()
        ax.plot(s_.month, s_[TARGET], color=INK, lw=0.8, marker="o", ms=1.4, label="Observed")
        ax.plot(q.month, q["Random Forest"], color=MC["Random Forest"], lw=0.9, label="Random Forest")
        ax.plot(q.month, q["Extra Trees"], color=MC["Extra Trees"], lw=0.9, label="Extra Trees")
        ax.plot(q.month, q["Well mean (training records)"], color="#52514e", lw=1.0, ls="--", label="Well training mean")
        ax.plot(q.month, q["Last observation (persistence)"], color=INK, lw=1.0, ls=":", label="Last observation")
        ax.axvline(last_train, color=MUTED, ls="-", lw=0.5); ax.invert_yaxis(); ax.set_ylabel("Depth to water (m)")
        import matplotlib.dates as mdates
        loc_ = mdates.AutoDateLocator(minticks=3, maxticks=6); ax.xaxis.set_major_locator(loc_)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(f"$\\bf{{({'abcdef'[k]})}}$  {NAME_FIX.get(wt.name[wid], wt.name[wid].title())}\nTest RMSE: Extra Trees {rm[wid]:.1f} m; last observation "
                     f"{rp_[wid]:.1f} m; skill {mfmt(skill[wid], '.2f')}", fontsize=6.3, loc="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965)); save(fig, "Figure8_hydrographs.png")
    json.dump({"wells": [{"well_id": w, "name": wt.name[w], "rmse_extra_trees": float(rm[w]), "rmse_last_observation": float(rp_[w]),
                          "skill_vs_last": float(skill[w]), "quantile": q} for w, q in zip(picks, qs)],
               "eligible_wells": int(len(elig)), "eligible_wells_extra_trees_better": int((skill > 0).sum()),
               "selection": "wells with >= 60 well-months and >= 15 test well-months (per-well chronological design, seed 42), ranked by "
                            "the skill of Extra Trees against the last observation, picked at quantiles " + str(qs)},
              open(FIG / "Figure8_wells.json", "w"), indent=1)

BASIN_LABEL = {"RIO MAIPO": "Río Maipo", "RIO RAPEL": "Río Rapel", "RIO ACONCAGUA": "Río Aconcagua", "RIO LIMARI": "Río Limarí",
               "RIO COPIAPO": "Río Copiapó", "RIO ELQUI": "Río Elqui", "RIO HUASCO": "Río Huasco", "RIO LOS CHOROS": "Río Los Choros",
               "COSTERAS R.ELQUI-R.LIMARI": "Coastal, Elqui–Limarí", "COSTERAS ACONCAGUA-MAIPO": "Coastal, Aconcagua–Maipo"}

def figS(t, m):
    s = pd.read_csv(OUT / "screen_random_split.csv").sort_values("RMSE", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.8), sharey=True)
    for ax, (met, lab) in zip(axes, METS):
        ax.barh(s.model, s[met], color=MC["Decision Tree"], height=0.6); ax.set_xlabel(lab)
        for yv, v in zip(s.model, s[met]): ax.text(v, yv, f" {v:.2f}" if met == "R2" else f" {v:.1f}", va="center", fontsize=6, color=INK2)
        panel(ax, "abc"[list(axes).index(ax)])
    fig.tight_layout(); save(fig, "FigureS_screen.png")

    top = t.Basin.value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.boxplot([t.loc[t.Basin == b, TARGET] for b in top], showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": MC["Decision Tree"]}, medianprops={"color": INK},
               whiskerprops={"color": INK2}, capprops={"color": INK2})
    ax.set_xticks(range(1, 11)); ax.set_xticklabels([BASIN_LABEL.get(b, b.title()) for b in top], rotation=35, ha="right"); ax.set_ylabel("Depth to water (m)")
    fig.tight_layout(); save(fig, "FigureS_basins.png")

    imp = pd.read_csv(OUT / "importance_permutation.csv")
    from matplotlib.ticker import MaxNLocator
    def pretty(u):
        u = u.replace("group:", "")
        if u.startswith("wclim_bio_bio"): return "BIO" + str(int(u[13:15]))
        return {"cop_dem_30_DEM_value": "Copernicus DEM", "elevation_NASADEM": "NASADEM elevation", "slope_NASADEM": "Slope",
                "aspect_NASADEM": "Aspect", "elevation_Alos_Palsar": "AW3D30 elevation", "Longitude_GCS_WGS_1984": "Longitude",
                "Latitude_GCS_WGS_1984": "Latitude", "land_cover_igbp": "Land cover", "climate_longterm": "Climate normals",
                "climate_monthly": "Monthly climate", "topography": "Topography", "land_cover": "Land cover",
                "location": "Coordinates", "tc_pr": "Precipitation", "tc_tmmn": "Min. temperature", "tc_tmmx": "Max. temperature",
                "tc_aet": "Actual ET", "tc_pet": "Reference ET", "tc_def": "Water deficit", "tc_pdsi": "PDSI",
                "tc_srad": "Shortwave radiation", "tc_vpd": "VPD", "tc_vs": "Wind speed"}[u]
    fig, axes = plt.subplots(2, 5, figsize=(W, 5.0), gridspec_kw={"height_ratios": [1, 2]})
    for k, d in enumerate(DESIGNS):
        gq = imp[(imp.design == d) & imp.is_group].sort_values("r2_drop_mean")
        axes[0, k].barh([pretty(u) for u in gq.unit], gq.r2_drop_mean, xerr=gq.r2_drop_sd,
                        color=MC["Random Forest"], height=0.6, error_kw={"elinewidth": 0.5})
        axes[0, k].set_title(DLAB[d], fontsize=6.8, pad=13); axes[0, k].tick_params(axis="y", labelsize=5.6)   # titles clear the row letters
        fq = imp[(imp.design == d) & ~imp.is_group].nlargest(10, "r2_drop_mean")[::-1]
        axes[1, k].barh([pretty(u) for u in fq.unit], fq.r2_drop_mean, color=MC["Random Forest"], height=0.6)
        axes[1, k].tick_params(axis="y", labelsize=5.6); axes[1, k].set_xlabel("Drop in R²", fontsize=6.5)
        for ax in axes[:, k]:
            ax.xaxis.set_major_locator(MaxNLocator(2)); ax.tick_params(axis="x", labelsize=6); ax.axvline(0, color=INK2, lw=0.5)
    panel(axes[0, 0], "a"); panel(axes[1, 0], "b")
    fig.tight_layout(w_pad=0.4); save(fig, "FigureS_importance.png")

    tm = m[m.model.isin(MODELS)].groupby(["design", "model"])[["R2", "train_R2"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(W, 2.8)); x = np.arange(len(DESIGNS))
    for j, mdl in enumerate(MODELS):
        q = tm[tm.model == mdl].set_index("design").reindex(DESIGNS)
        ax.scatter(x + (j - 1) * 0.18, q.train_R2, marker="^", s=18, color=MC[mdl], label=f"{mdl}, training")
        ax.scatter(x + (j - 1) * 0.18, q.R2, marker="o", s=18, color=MC[mdl], edgecolor=INK, lw=0.4, label=f"{mdl}, testing")
    ax.set_xticks(x); ax.set_xticklabels([DLAB[d] for d in DESIGNS]); ax.set_ylabel("R²"); ax.axhline(0, color=INK2, lw=0.6)
    ax.legend(ncol=3, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(); save(fig, "FigureS_train_test.png")

    f = OUT / "tuning.csv"
    if f.exists():
        tu = pd.read_csv(f); schemes = ["Default", "5-fold (shuffled)", "Forward chaining (5 folds)"]
        sc = {"Default": BASE_C, "5-fold (shuffled)": MC["Random Forest"], "Forward chaining (5 folds)": MC["Decision Tree"]}
        base = pd.read_csv(OUT / "metrics_main.csv")
        fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
        for c, d in enumerate(["temporal_per_well", "temporal_global"]):
            ax = axes[c]
            for j, sch in enumerate(schemes):
                q = tu[(tu.design == d) & (tu.scheme == sch)].set_index("model").reindex(MODELS)
                xs = np.arange(3) + (j - 1) * 0.27
                ax.bar(xs, q.RMSE, width=0.25, color=sc[sch], label=f"Test RMSE, {sch.lower() if sch != 'Default' else 'default settings'}")
                if sch != "Default":
                    ax.scatter(xs, q.cv_RMSE, marker="D", s=16, facecolor="white", edgecolor=INK, lw=0.8, zorder=3,
                               label="Cross-validation estimate of the selected configuration" if j == 1 else None)
            pers = base[(base.design == d) & (base.model == "Last observation (persistence)")].RMSE.mean()
            ax.axhline(pers, color=INK, ls=":", lw=1.0, label="Last observation (test RMSE)")
            ax.set_xticks(range(3)); ax.set_xticklabels([mm.replace(" ", "\n") for mm in MODELS], fontsize=6.8)
            ax.set_title(DLAB[d].replace(chr(10), " "), fontsize=7.5); panel(ax, "ab"[c])
        axes[0].set_ylabel("RMSE (m)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.84)); save(fig, "FigureS_tuning.png")

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
    sp = pd.read_csv(OUT / "splits_main.csv")
    for ax, d in zip(axes, ["temporal_per_well", "temporal_global"]):
        p = preds(d); yr = t.month.dt.year; te = np.isin(np.arange(len(t)), p.row.to_numpy())
        if d == "temporal_global":
            origin = pd.Timestamp(sp[sp.design == d].origin.iloc[0]); tr_mask = (t.month < origin).to_numpy()
        else:
            tr_mask = ~te
        ex = ~tr_mask & ~te
        data, cols, labs = [yr[tr_mask], yr[te]], [MC["Decision Tree"], MC["Random Forest"]], ["Training", "Testing"]
        if ex.any(): data.append(yr[ex]); cols.append(BASE_C); labs.append("Excluded (wells first monitored after the origin)")
        ax.hist(data, bins=np.arange(yr.min(), yr.max() + 2), stacked=True, color=cols, label=labs, edgecolor="white", linewidth=0.2)
        ax.set_title(DLAB[d].replace("\n", " ")); ax.set_xlabel("Year"); panel(ax, "ab"[list(axes).index(ax)])
    axes[0].set_ylabel("Well-months")
    h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "FigureS_partition.png")

    ABL = [("main", "All predictors", MC["Decision Tree"]), ("nocoord", "Without coordinates", "#86b6ef"),
           ("nomonthly", "Without monthly climate", MC["Extra Trees"]), ("lononly", "Longitude only", MC["Random Forest"])]
    frames = {tag: (m if tag == "main" else pd.read_csv(OUT / f"metrics_{tag}.csv")) for tag, _, _ in ABL if tag == "main" or (OUT / f"metrics_{tag}.csv").exists()}
    bb = best_baseline(m)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True); x = np.arange(len(DESIGNS)); wbar = 0.19
    for ax, mdl in zip(axes, ["Random Forest", "Extra Trees"]):
        for j, (tag, lab, col) in enumerate(ABL):
            if tag not in frames: continue
            fr = frames[tag]; vals = [fr[(fr.design == d) & (fr.model == mdl)].RMSE.mean() for d in DESIGNS]
            errs = [fr[(fr.design == d) & (fr.model == mdl)].RMSE.std() for d in DESIGNS]
            ax.bar(x + (j - 1.5) * (wbar + 0.01), vals, width=wbar, color=col, label=lab, yerr=errs,
                   error_kw={"elinewidth": 0.5, "capsize": 1.2, "ecolor": INK2})
        ref = [m[(m.design == d) & (m.model == bb[d])].RMSE.mean() for d in DESIGNS]
        ax.scatter(x, ref, marker="_", s=260, color=INK, lw=1.6, zorder=4, label="Reference baseline")
        ax.set_xticks(x); ax.set_xticklabels([DLAB[d].replace("Spatial block", "Spatial\nblock").replace("Chronological", "Chrono-\nlogical")
                                              for d in DESIGNS], fontsize=6.3); ax.set_title(mdl, fontsize=7.5)
        panel(ax, "ab"[list(axes).index(ax)])
    axes[0].set_ylabel("Testing RMSE (m)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "FigureS_ablation.png")

if __name__ == "__main__":
    import sys; sys.path.insert(0, str(HERE))
    t = pd.read_parquet(OUT / "model_table.parquet"); m = pd.read_csv(OUT / "metrics_main.csv"); g = shac()
    for fn in (lambda: fig02(t, g), lambda: fig03(t, g), lambda: fig04(m), lambda: fig05(m), fig06,
               lambda: fig07(t, g), lambda: fig08(t), lambda: figS(t, m)):
        try: fn()
        except FileNotFoundError as e: print("skipped (output not ready):", e)
