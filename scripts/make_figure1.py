"""
Figure 1: workflow diagram. Counts are read from the pipeline outputs (data_provenance.json, run_manifest.json),
so the diagram cannot drift from the analysis.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HEAD, BODY, EDGE = "#256abf", "#f4f8fd", "#86b6ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600})

prov = json.load(open(OUT / "data_provenance.json"))["counts"]
man = json.load(open(OUT / "run_manifest.json"))
g = {k: len(v) for k, v in man["predictor_groups"].items()}
fmt = lambda x: f"{x:,}"

stages = [
    ("1  Data", ["National DTW", "compilation", f"*{fmt(prov['raw_rows'])} records", "",
                 "Quality control: static", "readings only; missing,", "negative and flagged", "values removed", "",
                 f"*{fmt(prov['wells'])} physical wells", f"*{fmt(prov['well_months'])} well-months",
                 f"*{prov['date_first'][:4]}–{prov['date_last'][:4]}"]),
    ("2  Predictors", [f"*{man['n_features']} predictors,", "declared in advance", "",
                       f"Topography ({g['topography']})", f"Climate normals ({g['climate_longterm']})",
                       f"Monthly climate ({g['climate_monthly']}),", "re-extracted for each", "record's own month",
                       f"Land cover ({g['land_cover']})", f"Coordinates ({g['location']})", "",
                       "Identifiers, dates and", "metadata excluded"]),
    ("3  Evaluation", ["Decision Tree, Random", "Forest, Extra Trees;", "imputation fitted on", "training data only", "",
                       "*Five splitting designs", "· random", "· well-based", "· spatial block", "· chronological, per well",
                       "· chronological, global", "  origin", "", "Five seeds per design;", "admissible baselines on", "the same partitions:", "training mean, well", "training mean, temporal", "and neighbor", "interpolation, last", "observation"]),
    ("4  Analyses", ["Grouped permutation", "importance", "", "Drop-column ablations:", "coordinates, monthly", "climate, longitude only", "",
                     "Error by horizon and", "by latitude band", "",
                     "Sensitivity: record", "length, training", "fraction, origin date", "",
                     "Tuning with shuffled", "vs. forward-chaining", "folds"]),
    ("5  Diagnostics", ["Good / Poor labels", "(absolute error ≤ 5 m)", "", "Per-well RMSE and", "skill maps", "",
                        "Observed and predicted", "hydrographs"]),
]

LH, GAP_H, FS = 0.155, 0.075, 6.2
need = max(sum(LH if ln else GAP_H for ln in lines) for _, lines in stages)
W = 6.73; hh = 0.28; H = need + hh + 0.30
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
n = len(stages); gap = 0.17; bw = (W - 0.04 - gap * (n - 1)) / n; top, bottom = H - 0.03, 0.03
for i, (title, lines) in enumerate(stages):
    x = 0.02 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, bottom), bw, top - bottom, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=BODY, ec=EDGE, lw=0.8))
    ax.add_patch(FancyBboxPatch((x, top - hh), bw, hh, boxstyle="round,pad=0,rounding_size=0.06", fc=HEAD, ec=HEAD, lw=0.8))
    ax.text(x + bw / 2, top - hh / 2, title, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    y = top - hh - 0.12
    for ln in lines:
        if ln:
            bold = ln.startswith("*"); ln = ln.lstrip("*")
            ax.text(x + 0.07, y, ln, ha="left", va="top", color=INK if bold else INK2, fontsize=FS,
                    fontweight="bold" if bold else "normal")
        y -= LH if ln else GAP_H
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.015, (top + bottom) / 2), (x + bw + gap - 0.015, (top + bottom) / 2),
                                     arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.0))
# guard: no text may cross its box edge
fig.set_dpi(600); fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()   # measured at the output resolution
for t in ax.texts:
    bb = inv.transform(t.get_window_extent(r)); xs = bb[:, 0]
    col = int((xs[0] - 0.02) // (bw + gap)); x0 = 0.02 + col * (bw + gap)
    assert xs[1] <= x0 + bw - 0.06, f"overflow: {t.get_text()}"
fig.savefig(FIG / "Figure1_workflow.png", facecolor="white"); plt.close(fig); print("wrote Figure1_workflow.png")
