"""
Derive every number reported in the manuscript (outputs/facts.json) and every table (outputs/tables/*.csv) from the
pipeline outputs. Labels are typed here; numbers are not. Run after rift_pipeline.py and make_figures.py:

    python make_facts.py

Number formatting: typographic minus, no negative zero; means and standard deviations are over the five repeats
(sample SD); skill scores SS = 1 - MSE/MSE_ref are computed seed by seed on identical partitions.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import rift_pipeline as rp

OUT = rp.OUT; FIG = HERE / "figures"; TAB = OUT / "tables"; TAB.mkdir(exist_ok=True)
MODELS = ["Decision Tree", "Random Forest", "Extra Trees"]
SHORT = {"Decision Tree": "dt", "Random Forest": "rf", "Extra Trees": "et"}
DESIGNS = ["random", "well", "spatial", "temporal_per_well", "temporal_global"]
DNAME = {"random": "Random", "well": "Well-based", "spatial": "Spatial block", "temporal_per_well": "Chronological (per well)",
         "temporal_global": "Chronological (global origin)"}
BASE = {"Training mean": ("base_mean", "Training mean"),
        "Well mean (training records)": ("base_wellmean", "Well training mean"),
        "Last observation (persistence)": ("base_last", "Last observation"),
        "Neighbor interpolation (k=5, IDW)": ("base_nbr", "Neighbor interpolation"),
        "Temporal interpolation (same well)": ("base_tint", "Temporal interpolation")}
KEY = {**SHORT, **{k: v[0] for k, v in BASE.items()}}
LABEL = {**{m_: m_ for m_ in MODELS}, **{k: v[1] for k, v in BASE.items()}}
TARGET, WELL = rp.TARGET, rp.WELL
F = {}

def _minus(txt):
    if txt.lstrip("-").strip("0.,") == "": txt = txt.lstrip("-")
    return txt.replace("-", "−")
def n(x): return _minus(f"{int(round(float(x))):,}")
def f1(x): return _minus(f"{x:.1f}")
def f2(x): return _minus(f"{x:.2f}")
def f3(x): return _minus(f"{x:.3f}")
def lat_s(x): return f"{abs(x):.1f}° S"
def write(name, rows):
    with open(TAB / name, "w", newline="", encoding="utf-8") as fh: csv.writer(fh).writerows(rows)
def opt(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None
def rng(a, b): return a if a == b else f"{a}–{b}"

# ======================================================================== data
prov = json.load(open(OUT / "data_provenance.json")); c = prov["counts"]
for k in ["raw_rows", "status_blank", "target_present", "non_negative", "outlier_flag_false", "well_months", "wells",
          "code_merges", "colocation_groups", "well_months_from_multiple_records"]:
    F[f"data_{k}"] = n(c[k])
F["data_removed_status"] = n(c["raw_rows"] - c["status_blank"]); F["data_removed_nan"] = n(c["status_blank"] - c["target_present"])
F["data_removed_negative"] = n(c["target_present"] - c["non_negative"]); F["data_removed_outlier"] = n(c["non_negative"] - c["outlier_flag_false"])
F["data_year_first"] = c["date_first"][:4]; F["data_year_last"] = c["date_last"][:4]
F["data_sha256_short"] = prov["sha256"][:12]; F["data_max_spread_km"] = f2(c["max_within_well_spread_km"])

t = pd.read_parquet(OUT / "model_table.parquet"); y = t[TARGET]
F.update({"dtw_mean": f1(y.mean()), "dtw_median": f1(y.median()), "dtw_sd": f1(y.std()), "dtw_max": f1(y.max()),
          "dtw_q25": f1(y.quantile(.25)), "dtw_q75": f1(y.quantile(.75)), "dtw_n_lt1m": n((y < 1).sum()),
          "dtw_pct_lt1m": f1(100 * (y < 1).mean()), "dtw_n_gt50": n((y > 50).sum())})
wmean = t.groupby(WELL)[TARGET].transform("mean")
F["between_well_var_pct"] = f1(100 * ((wmean - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())
per_well = t.groupby(WELL).size()
F.update({"wellmonths_per_well_median": n(per_well.median()), "wellmonths_per_well_max": n(per_well.max()),
          "wells_le3": n((per_well <= 3).sum()), "wells_ge27": n((per_well >= 27).sum())})
b = t.groupby("Basin")[TARGET].agg(["size", "median"])
for key, name in [("maipo", "RIO MAIPO"), ("aconcagua", "RIO ACONCAGUA")]:
    F[f"basin_{key}_n"] = n(b.loc[name, "size"]); F[f"basin_{key}_median"] = f1(b.loc[name, "median"])
F["data_basins"] = n(t["Basin"].nunique())
F["pct_wm_30_34S"] = f1(100 * t.well_lat.between(-34, -30).mean())
F["pct_wm_north_of_26S"] = f1(100 * (t.well_lat > -26).mean()); F["pct_wm_south_of_38S"] = f1(100 * (t.well_lat < -38).mean())
F["well_lat_north"] = lat_s(t.well_lat.max()); F["well_lat_south"] = lat_s(t.well_lat.min())
F["well_lon_west"] = f"{abs(t.well_lon.min()):.1f}° W"; F["well_lon_east"] = f"{abs(t.well_lon.max()):.1f}° W"
yc = t.month.dt.year.value_counts(); F["wm_per_year_min"] = n(yc.min()); F["wm_per_year_max"] = n(yc.max())

man = json.load(open(OUT / "run_manifest.json"))
feats = man["features"]; F["n_predictors"] = str(man["n_features"]); F["python_version"] = man["python"]
F["n_env_predictors"] = str(man["n_features"] - len(man["predictor_groups"]["location"]))
F["n_predictors_nocoord"] = str(len(rp.FEATURES_NOCOORD)); F["n_predictors_nomonthly"] = str(len(rp.FEATURES_NOMONTHLY))
for pk, pv in man["packages"].items(): F[f"ver_{pk.replace('-', '_')}"] = pv
import geopandas, pyarrow
F["ver_geopandas"] = geopandas.__version__; F["ver_pyarrow"] = pyarrow.__version__
F["n_missing_predictor_values"] = n(int(t[feats].isna().sum().sum()))
PRETTY = {"Longitude_GCS_WGS_1984": "longitude", "Latitude_GCS_WGS_1984": "latitude", "elevation_NASADEM": "NASADEM elevation",
          "cop_dem_30_DEM_value": "Copernicus DEM elevation", "elevation_Alos_Palsar": "AW3D30 elevation", "land_cover_igbp": "land cover",
          "slope_NASADEM": "slope", "aspect_NASADEM": "aspect"}
pretty = lambda col: PRETTY.get(col, ("BIO" + str(int(col[13:15]))) if col.startswith("wclim_bio_bio") else col)
r_ = t[feats].corrwith(y).abs().sort_values(ascending=False)
F["corr_max_abs"] = f2(r_.iloc[0]); F["corr_top1_name"] = pretty(r_.index[0]); F["corr_top2_name"] = pretty(r_.index[1]); F["corr_top2_abs"] = f2(r_.iloc[1])
sig = t.groupby(WELL)[rp.STATIC_RAW + [rp.LON, rp.LAT]].first().round(6)
F["wells_unique_static_signature"] = n((~sig.duplicated(keep=False)).sum())
lon_w = t.groupby(WELL)[rp.LON].first().round(6)
F["wells_unique_longitude"] = n((~lon_w.duplicated(keep=False)).sum()); F["wells_distinct_longitudes"] = n(lon_w.nunique())

# raw-file checks (read from the source file)
raw = pd.read_csv(rp.CSV, encoding="latin-1", low_memory=False,
                  usecols=["Name", "Code", rp.LON, rp.LAT, "Status", TARGET, "Outlier", "Date_String", "terraclim_rs_date",
                           "mod16_et_rs_date", "modis_lc_LC_Type1_value", "modis_lc_rs_date"])
F["data_names"] = n(raw.Name.nunique())
st = raw.Status.fillna("").str.strip()
for k, v in {"dynamic": "Din", "noaccess": "Sin Acceso", "dry": "Seco", "silted": "Embancado", "flowing": "Surgente"}.items():
    F[f"status_{k}"] = n(st.str.startswith(v).sum())
rec_m = pd.to_datetime(raw.Date_String).dt.to_period("M"); gee_m = pd.to_datetime(raw.terraclim_rs_date, errors="coerce").dt.to_period("M")
F["tc_gee_month_mismatch_pct"] = f1(100 * (gee_m != rec_m)[gee_m.notna()].mean())
mod = pd.to_datetime(raw.mod16_et_rs_date, errors="coerce"); mm = mod.mode().iloc[0]
F["mod16_modal_date"] = f"{mm.day} {mm.strftime('%B %Y')}"; F["mod16_modal_date_pct"] = f1(100 * (mod == mm).mean())
lc = raw[raw.Status.isna() & raw[TARGET].notna() & (raw[TARGET] >= 0) & ~raw.Outlier.astype(bool)].copy()
lc, _, _ = rp.build_well_keys(lc)
k_ = lc.groupby(WELL)["modis_lc_LC_Type1_value"].nunique()
F["lc_wells_multi_class"] = n((k_ > 1).sum()); F["lc_wells_multi_class_pct"] = f1(100 * (k_ > 1).mean())
mode_ = lc.groupby(WELL)["modis_lc_LC_Type1_value"].agg(lambda s: s.mode().iloc[0])
F["lc_records_equal_mode_pct"] = f1(100 * (lc[WELL].map(mode_) == lc["modis_lc_LC_Type1_value"]).mean())
lcy = pd.to_datetime(lc["modis_lc_rs_date"], errors="coerce").dt.year; recy = pd.to_datetime(lc["Date_String"]).dt.year
F["lc_pct_records_same_year"] = f1(100 * (lcy == recy).mean()); F["lc_pct_records_2023"] = f1(100 * (lcy == 2023).mean())
F["lc_pct_records_pre2001"] = f1(100 * (recy < 2001).mean()); F["lc_product_year_max"] = str(int(lcy.max())); F["lc_product_year_min"] = str(int(lcy.min()))
F["lc_pre2001_product_years"] = " or ".join(str(int(v)) for v in sorted(lcy[recy < 2001].dropna().unique()))
pre = lcy[recy < 2001]
assert set(pre.dropna().astype(int)) == {int(lcy.min()), int(lcy.max())}, "pre-2001 records carry only the first and last products"
F["lc_pre2001_pct_last_product"] = f1(100 * (pre == lcy.max()).mean())
F["lc_n_records_filtered"] = n(len(lc))
tcv = json.load(open(OUT / "terraclimate_refetch_log.json"))
F["tc_r_min"] = f2(min(tcv["pearson_r"].values())); F["tc_r_max"] = f3(max(tcv["pearson_r"].values()))
F["tc_rows_compared"] = n(tcv["rows_compared"])
bx = tcv["bbox"]; F["tc_box"] = f"{abs(bx['north']):.1f}–{abs(bx['south']):.1f}° S, {abs(bx['east']):.0f}–{abs(bx['west']):.0f}° W"
F["tc_years"] = f"{tcv['years'][0]}–{tcv['years'][1]}"

# ======================================================================== splits
sp = pd.read_csv(OUT / "splits_main.csv")
for d in DESIGNS:
    s = sp[sp.design == d]
    for col, key, fmt in [("n_train", "ntrain", n), ("n_test", "ntest", n), ("pct_train_of_table", "pcttrain", f1),
                          ("pct_test_of_table", "pcttest", f1), ("wells_test", "wellstest", n), ("wells_train", "wellstrain", n),
                          ("test_wells_also_in_train", "shared_wells", n), ("n_excluded", "excluded", n)]:
        F[f"{d}_{key}_min"], F[f"{d}_{key}_max"] = fmt(s[col].min()), fmt(s[col].max())
    F[f"{d}_excluded"] = F[f"{d}_excluded_max"]
F["pcttest_all_min"] = f1(sp.pct_test_of_table.min()); F["pcttest_all_max"] = f1(sp.pct_test_of_table.max())
for d in ["temporal_per_well", "temporal_global"]:
    s = sp[sp.design == d].iloc[0]
    F[f"{d}_h_median"] = f1(s.horizon_median_y); F[f"{d}_h_q25"] = f1(s.horizon_q25_y); F[f"{d}_h_q75"] = f1(s.horizon_q75_y)
F["temporal_per_well_before_last_train_pct"] = f1(sp[sp.design == "temporal_per_well"].test_before_last_train_month_pct.iloc[0])
F["temporal_global_origin_text"] = pd.Timestamp(sp[sp.design == "temporal_global"].origin.iloc[0]).strftime("%B %Y")
F["temporal_global_excluded_pct"] = f1(100 * sp[sp.design == "temporal_global"].n_excluded.iloc[0] / len(t))
trg, teg, ginfo = rp.split_temporal_global(t)
F["temporal_global_last_train_text"] = pd.Timestamp(ginfo["last_train_month"]).strftime("%B %Y")
# global-origin horizons run from the partition's last training month; how much older is each test well's own last training month?
lt_w = t.iloc[trg].groupby(WELL).month.max(); te_wells = pd.Index(t.iloc[teg][WELL].unique())
gap_y = (pd.Timestamp(ginfo["last_train_month"]) - lt_w.reindex(te_wells)).dt.days / 365.25
F["temporal_global_wells_last_train_gt1y"] = n((gap_y > 1).sum()); F["temporal_global_n_test_wells_all"] = n(len(te_wells))
wl_all = t.groupby(WELL).well_lat.first()
test_w = set(t.iloc[teg][WELL]); train_w = set(t.iloc[trg][WELL]); excl_w = set(wl_all.index) - train_w
F["temporal_global_excluded_wells"] = n(len(excl_w))
F["temporal_global_test_lat_south"] = lat_s(wl_all.loc[list(test_w)].min())
F["temporal_global_wells_south36"] = n((wl_all < -36).sum()); F["temporal_global_test_wells_south36"] = n((wl_all.loc[list(test_w)] < -36).sum())
first_month_south = t[t.well_lat < -36].groupby(WELL).month.min()
F["south36_first_year_min"] = str(first_month_south.min().year); F["south36_first_year_max"] = str(first_month_south.max().year)
s = sp[sp.design == "spatial"]
F["spatial_cell_deg_lon"] = f2(s.cell_deg_lon.iloc[0]); F["spatial_cell_deg_lat"] = f2(s.cell_deg_lat.iloc[0])
midlat = np.radians(t.well_lat.mean())
F["spatial_cell_km_ew"] = n(s.cell_deg_lon.iloc[0] * 111.32 * np.cos(midlat)); F["spatial_cell_km_ns"] = n(s.cell_deg_lat.iloc[0] * 110.57)
F["spatial_occupied_cells"] = n(s.occupied_cells.iloc[0]); F["spatial_test_cells_min"] = n(s.test_cells.min()); F["spatial_test_cells_max"] = n(s.test_cells.max())
F["spatial_nn_km_median_min"] = f1(s.test_to_nearest_train_km_median.min()); F["spatial_nn_km_median_max"] = f1(s.test_to_nearest_train_km_median.max())
F["spatial_nn_km_min"] = f1(s.test_to_nearest_train_km_min.min())

# ======================================================================== main metrics, baselines and skill
m = pd.read_csv(OUT / "metrics_main.csv")
for (d, mdl), q in m.groupby(["design", "model"]):
    key = KEY[mdl]
    for met, fmt in [("R2", f2), ("RMSE", f1), ("MAE", f1), ("WAPE", f1), ("MdAPE", f1), ("MAPE", f1), ("MAPE_ge1m", f1)]:
        F[f"{d}_{key}_{met}"] = fmt(q[met].mean()); F[f"{d}_{key}_{met}_sd"] = fmt(q[met].std(ddof=1)) if q.seed.nunique() > 1 else "0"
    F[f"{d}_{key}_RMSE2"] = f2(q.RMSE.mean())
tr_r2 = m[m.model.isin(MODELS)].train_R2
F["trainR2_min"] = f3(tr_r2.min()); F["trainR2_rf_min"] = f3(m[m.model == "Random Forest"].train_R2.min())
F["trainR2_dt_et_min"] = f3(m[m.model.isin(["Decision Tree", "Extra Trees"])].train_R2.min())

def seed_rmse(frame, d, mdl): return frame[(frame.design == d) & (frame.model == mdl)].set_index("seed").RMSE
def skill(frame, d, mdl, ref, ref_frame=None):
    a = seed_rmse(frame, d, mdl); r = seed_rmse(ref_frame if ref_frame is not None else frame, d, ref).reindex(a.index)
    return 1 - (a / r) ** 2
REF = {}
for d in DESIGNS:
    bmean = m[(m.design == d) & m.model.isin(BASE)].groupby("model").RMSE.mean()
    REF[d] = bmean.idxmin(); F[f"{d}_ref_baseline"] = LABEL[REF[d]].lower(); F[f"{d}_ref_RMSE"] = f1(bmean.min())
SK = {}
for d in DESIGNS:
    for mdl in MODELS:
        ss = skill(m, d, mdl, REF[d]); SK[(d, mdl)] = ss
        F[f"{d}_{SHORT[mdl]}_skill"] = f2(ss.mean()); F[f"{d}_{SHORT[mdl]}_skill_sd"] = f2(ss.std(ddof=1))
        F[f"{d}_{SHORT[mdl]}_skill_npos"] = str(int((ss > 0).sum())); F[f"{d}_{SHORT[mdl]}_skill_min"] = f2(ss.min()); F[f"{d}_{SHORT[mdl]}_skill_max"] = f2(ss.max())
        for bname in BASE:
            if ((m.design == d) & (m.model == bname)).any():
                sb_ = skill(m, d, mdl, bname)
                F[f"{d}_{SHORT[mdl]}_skill_vs_{BASE[bname][0]}"] = f2(sb_.mean())
                F[f"{d}_{SHORT[mdl]}_skill_vs_{BASE[bname][0]}_npos"] = str(int((sb_ > 0).sum()))
    best = max(MODELS, key=lambda x: SK[(d, x)].mean()); F[f"{d}_skill_best_model"] = best; F[f"{d}_skill_best"] = f2(SK[(d, best)].mean())
ss_sp = SK[("spatial", "Extra Trees")]; pos_, neg_ = ss_sp[ss_sp > 0], ss_sp[ss_sp <= 0]
F["spatial_et_skill_pos_min"], F["spatial_et_skill_pos_max"] = f2(pos_.min()), f2(pos_.max())
F["spatial_et_skill_neg_text"] = " and ".join(f2(v) for v in sorted(neg_, reverse=True)); F["spatial_et_skill_nneg"] = str(len(neg_))
# largest seed-mean skill of any learner against the well training mean under the chronological designs
F["chrono_ens_skill_vs_wellmean_max_abs"] = f2(max(abs(skill(m, d, mdl, "Well mean (training records)").mean())
                                                   for d in ["temporal_per_well", "temporal_global"] for mdl in ["Random Forest", "Extra Trees"]))
# baselines against the design's other baselines (e.g. neighbor interpolation vs training mean in the spatial design)
for d in ["well", "spatial"]:
    ss = skill(m, d, "Neighbor interpolation (k=5, IDW)", "Training mean")
    F[f"{d}_base_nbr_skill_vs_base_mean"] = f2(ss.mean()); F[f"{d}_base_nbr_skill_vs_base_mean_npos"] = str(int((ss > 0).sum()))

T3 = [["Design", "Model or baseline", "R²", "RMSE (m)", "WAPE (%)", "Skill score"]]
order_b = list(BASE)
for d in DESIGNS:
    first = True
    for mdl in MODELS + [bn for bn in order_b if ((m.design == d) & (m.model == bn)).any()]:
        q = m[(m.design == d) & (m.model == mdl)]
        multi = q.seed.nunique() > 1 and q.RMSE.std() > 1e-9
        sd = lambda col, fmt: (" ± " + fmt(q[col].std(ddof=1))) if multi else ""
        if mdl in MODELS:
            sk = SK[(d, mdl)]; sks = f2(sk.mean()) + (" ± " + f2(sk.std(ddof=1)) if multi else "")
        else:
            sks = "reference" if mdl == REF[d] else ""
        T3.append([DNAME[d] if first else "", LABEL[mdl], f2(q.R2.mean()) + sd("R2", f2), f2(q.RMSE.mean()) + sd("RMSE", f2),
                   f1(q.WAPE.mean()) + sd("WAPE", f1), sks])
        first = False
write("Table3.csv", T3)

# Table S (all error measures)
TS = [["Design", "Model or baseline", "R²", "RMSE (m)", "MAE (m)", "WAPE (%)", "MdAPE (%)", "MAPE (%)", "MAPE, DTW ≥ 1 m (%)", "Test well-months"]]
for d in DESIGNS:
    first = True
    for mdl in MODELS + [bn for bn in order_b if ((m.design == d) & (m.model == bn)).any()]:
        q = m[(m.design == d) & (m.model == mdl)]; multi = q.seed.nunique() > 1 and q.RMSE.std() > 1e-9
        ms_ = lambda col, fmt: fmt(q[col].mean()) + ((" ± " + fmt(q[col].std(ddof=1))) if multi else "")
        TS.append([DNAME[d] if first else "", LABEL[mdl], ms_("R2", f2), ms_("RMSE", f2), ms_("MAE", f2), ms_("WAPE", f1),
                   ms_("MdAPE", f1), ms_("MAPE", f1), ms_("MAPE_ge1m", f1), rng(n(q.n_test.min()), n(q.n_test.max()))])
        first = False
write("TableS_metrics.csv", TS)

# Table 2 (designs)
T2 = [["Design", "Train–test dependence removed", "Question the estimate answers", "Training / testing well-months", "Test wells (of which also in training)"]]
text = {"random": ("None (within-well and between-well dependence kept)", "Interpolation among records similar to the training data"),
        "well": ("Within-well dependence (no test well, or co-located well, in training)", "Error at a new well inside the sampled domain"),
        "spatial": ("Within-well plus short-range between-well dependence (whole grid cells held out; no buffer)", "Extrapolation to unsampled regions"),
        "temporal_per_well": ("Later records of each well; other wells' concurrent records stay in training", "Later levels at monitored wells within a shared period"),
        "temporal_global": ("All information after one calendar origin", "Levels after a common origin at monitored wells, given observed climate")}
for d in DESIGNS:
    T2.append([DNAME[d], text[d][0], text[d][1],
               f"{rng(F[f'{d}_ntrain_min'], F[f'{d}_ntrain_max'])} / {rng(F[f'{d}_ntest_min'], F[f'{d}_ntest_max'])}",
               f"{rng(F[f'{d}_wellstest_min'], F[f'{d}_wellstest_max'])} ({rng(F[f'{d}_shared_wells_min'], F[f'{d}_shared_wells_max'])})"])
write("Table2.csv", T2)

# Table 1 (predictor sources; labels only)
write("Table1.csv", [["Category", "Product", "Variables used as predictors", "Native resolution", "Source"],
    ["Topography", "Copernicus DEM GLO-30", "Elevation", "30 m", "[@CopernicusDEMnd]"],
    ["Topography", "NASADEM", "Elevation, slope, aspect", "1 arc-second (~30 m)", "[@NASAJPL2020]"],
    ["Topography", "AW3D30 v3.2 (ALOS PRISM)", "Elevation", "1 arc-second (~30 m)", "[@Tadono2014; @Takaku2020]"],
    ["Long-term climate", "WorldClim 1.4", "19 bioclimatic variables (BIO1–BIO19)", "30 arc-seconds (~1 km)", "[@Hijmans2005]"],
    ["Monthly climate", "TerraClimate", "Precipitation, minimum and maximum temperature, actual and reference evapotranspiration, climatic water deficit, PDSI, downward shortwave radiation, vapor pressure deficit, wind speed", "1/24° (~4 km)", "[@Abatzoglou2018]"],
    ["Land cover", "MODIS MCD12Q1 v061", "IGBP class (per-well mode)", "500 m", "[@Friedl2022]"],
    ["Location", "Well coordinates", "Longitude, latitude", "Point", "[@VenegasQuinones2024; @VenegasQuinones2023]"]])

# ======================================================================== ablations
ABL = {"nocoord": "Without coordinates", "nomonthly": "Without monthly climate", "lononly": "Longitude only"}
TA = [["Design", "Learner", "All predictors: RMSE (m)", "Without coordinates", "Without monthly climate", "Longitude only"]]
abl = {tag: opt(f"metrics_{tag}.csv") for tag in ABL}
for d in DESIGNS:
    first = True
    for mdl in MODELS:
        row = [DNAME[d] if first else "", mdl, f2(seed_rmse(m, d, mdl).mean())]
        for tag, fr in abl.items():
            if fr is None: row.append("–"); continue
            a = seed_rmse(fr, d, mdl); row.append(f2(a.mean()))
            F[f"{tag}_{d}_{SHORT[mdl]}_RMSE"] = f2(a.mean()); F[f"{tag}_{d}_{SHORT[mdl]}_R2"] = f2(fr[(fr.design == d) & (fr.model == mdl)].R2.mean())
            F[f"{tag}_{d}_{SHORT[mdl]}_dRMSE"] = f2(round(a.mean(), 2) - round(seed_rmse(m, d, mdl).mean(), 2))   # as in Table S4
            ss = skill(fr, d, mdl, REF[d], ref_frame=m); F[f"{tag}_{d}_{SHORT[mdl]}_skill"] = f2(ss.mean())
            F[f"{tag}_{d}_{SHORT[mdl]}_skill_npos"] = str(int((ss > 0).sum()))
        TA.append(row); first = False
write("TableS_ablation.csv", TA)
for tag, fr in abl.items():
    if fr is None: continue
    # differences of the rounded means, so that the text agrees with Table S4
    dd = pd.Series({(d, mdl): round(seed_rmse(fr, d, mdl).mean(), 2) - round(seed_rmse(m, d, mdl).mean(), 2) for d in DESIGNS for mdl in MODELS})
    rt = dd[[x for x in dd.index if x[0] in ("random", "temporal_per_well", "temporal_global") and x[1] != "Decision Tree"]]
    F[f"{tag}_rt_ens_max_abs_dRMSE"] = f2(rt.abs().max())
    ch_ = dd[[x for x in dd.index if x[0] in ("temporal_per_well", "temporal_global") and x[1] != "Decision Tree"]]
    F[f"{tag}_chrono_ens_max_abs_dRMSE"] = f2(ch_.abs().max())
    rta = dd[[x for x in dd.index if x[0] in ("random", "temporal_per_well", "temporal_global")]]
    F[f"{tag}_rt_all_max_abs_dRMSE"] = f2(rta.abs().max())
    ens = dd[[x for x in dd.index if x[1] != "Decision Tree"]]
    F[f"{tag}_ens_max_dRMSE"] = f2(ens.max()); F[f"{tag}_ens_min_dRMSE"] = f2(ens.min())
if abl.get("nomonthly") is not None:   # does the small random-split gain of Random Forest over the well training mean survive without monthly climate?
    ss = skill(abl["nomonthly"], "random", "Random Forest", "Well mean (training records)", ref_frame=m)
    F["nomonthly_random_rf_skill_vs_base_wellmean"] = f2(ss.mean()); F["nomonthly_random_rf_skill_vs_base_wellmean_npos"] = str(int((ss > 0).sum()))

# ======================================================================== screen
sc = opt("screen_random_split.csv")
if sc is not None:
    for _, rr in sc.iterrows():
        k = rr.model.lower().replace(" ", "_").replace("-", "_")
        F[f"screen_{k}_R2"] = f2(rr.R2); F[f"screen_{k}_RMSE"] = f1(rr.RMSE)
    lin = sc[sc.model.isin(["Linear Regression", "Ridge", "Lasso", "Elastic Net"])]
    F["screen_linear_R2_min"], F["screen_linear_R2_max"] = f2(lin.R2.min()), f2(lin.R2.max())
    F["screen_best_model"] = sc.loc[sc.R2.idxmax(), "model"]; F["screen_best_R2"] = f2(sc.R2.max())

# ======================================================================== sensitivity (learners and baselines on identical partitions)
mr = opt("sensitivity_min_records.csv")
if mr is not None:
    q = mr[mr.model == "Random Forest"]; F["sens_mr_rf_R2_min"], F["sens_mr_rf_R2_max"] = f2(q.R2.min()), f2(q.R2.max())
    F["sens_mr_rf_RMSE_min"], F["sens_mr_rf_RMSE_max"] = f1(q.RMSE.min()), f1(q.RMSE.max())
for name, col, tag in [("sensitivity_ratio_per_well.csv", "train_fraction", "ratio"), ("sensitivity_origin_global.csv", "origin_percentile", "origin")]:
    df = opt(name)
    if df is None: continue
    for _, rr in df.iterrows():
        v = int(round(rr[col] * 100)) if col == "train_fraction" else int(rr[col])
        F[f"sens_{tag}{v}_{SHORT[rr.model]}_R2"] = f2(rr.R2); F[f"sens_{tag}{v}_{SHORT[rr.model]}_RMSE"] = f1(rr.RMSE)
        F[f"sens_{tag}{v}_horizon"] = f1(rr.horizon_median_y)
sbl = opt("sensitivity_baselines.csv")
if sbl is not None:
    for _, rr in sbl.iterrows():
        tag = {"min_records": "mr", "train_fraction": "ratio", "origin_percentile": "origin"}[rr.analysis]
        v = int(round(rr.value * 100)) if rr.analysis == "train_fraction" else int(rr.value)
        F[f"sens_{tag}{v}_{BASE[rr.model][0]}_RMSE"] = f1(rr.RMSE); F[f"sens_{tag}{v}_{BASE[rr.model][0]}_R2"] = f2(rr.R2)
    mrb = sbl[(sbl.analysis == "min_records") & (sbl.model == "Last observation (persistence)")]
    F["sens_mr_last_RMSE_min"], F["sens_mr_last_RMSE_max"] = f1(mrb.RMSE.min()), f1(mrb.RMSE.max())
    # partitions on which the better ensemble beats the last observation, and their median horizons
    rev = []
    for name, col, an in [("sensitivity_ratio_per_well.csv", "train_fraction", "train_fraction"), ("sensitivity_origin_global.csv", "origin_percentile", "origin_percentile")]:
        df = opt(name)
        if df is None: continue
        for v, g in df.groupby(col):
            last = sbl[(sbl.analysis == an) & np.isclose(sbl.value, v) & (sbl.model == "Last observation (persistence)")].RMSE.iloc[0]
            if g[g.model != "Decision Tree"].RMSE.min() < last: rev.append((an, v, g.horizon_median_y.iloc[0]))
    F["sens_reversal_n"] = str(len(rev)); F["sens_reversal_h_min"] = f1(min(r[2] for r in rev)); F["sens_reversal_h_max"] = f1(max(r[2] for r in rev))
    fr_ = sorted(r[1] for r in rev if r[0] == "train_fraction"); og_ = sorted(int(r[1]) for r in rev if r[0] == "origin_percentile")
    assert len(fr_) < 2 or np.allclose(np.diff(fr_), 0.1), "reversing training fractions are not contiguous"
    F["sens_reversal_fractions"] = rng(f"{fr_[0]:.1f}", f"{fr_[-1]:.1f}")
    F["sens_reversal_origins"] = " and ".join(f"{o}th" for o in og_); F["sens_reversal_n_origins"] = str(len(og_))
    og_all = sorted(int(v) for v in opt("sensitivity_origin_global.csv").origin_percentile.unique())
    assert og_ == og_all[:1], "the text says only the earliest split position reverses"

# ======================================================================== horizon
hz = opt("horizon_errors.csv")
if hz is not None:
    for _, rr in hz.iterrows():
        F[f"hz_{rr.design}_h{int(rr.h_lo)}_{KEY[rr.model]}_RMSE"] = f1(rr.RMSE); F[f"hz_{rr.design}_h{int(rr.h_lo)}_n"] = n(rr.n_test)
    pv = hz.pivot_table(index=["design", "h_lo"], columns="model", values="RMSE").sort_index()   # bins in horizon order
    F["hz_ens_wellmean_max_abs_diff"] = f1((pv[["Random Forest", "Extra Trees"]].sub(pv["Well mean (training records)"], axis=0)).abs().max().max())
    ens_best = pv[["Random Forest", "Extra Trees"]].min(axis=1)
    gap = ens_best - pv["Last observation (persistence)"]
    for d in ["temporal_per_well", "temporal_global"]:
        g_ = gap.loc[d]; lab_ = list(g_.index)
        F[f"hz_{d}_gap_first"] = f1(g_.iloc[0]); F[f"hz_{d}_gap_last"] = f1(g_.iloc[-1])
        F[f"hz_{d}_gap_max_before_last"] = f1(g_.iloc[:-1].max()); F[f"hz_{d}_gap_min_before_last"] = f1(g_.iloc[:-1].min())
        F[f"hz_{d}_last_better_bins"] = str(int((g_ > 0).sum())); F[f"hz_{d}_bins"] = str(len(g_))

# ======================================================================== tuning
tu = opt("tuning.csv")
if tu is not None:
    kf, fw = tu[tu.scheme == "5-fold (shuffled)"], tu[tu.scheme == "Forward chaining (5 folds)"]
    T4 = [["Design", "Learner", "Default: test RMSE", "Shuffled 5-fold: CV estimate", "Shuffled 5-fold: test RMSE",
           "Forward chaining: CV estimate", "Forward chaining: test RMSE"]]
    for d in ["temporal_per_well", "temporal_global"]:
        for i, mdl in enumerate(MODELS):
            g_ = tu[(tu.design == d) & (tu.model == mdl)].set_index("scheme")
            T4.append([DNAME[d] if i == 0 else "", mdl, f2(g_.loc["Default", "RMSE"]), f2(g_.loc["5-fold (shuffled)", "cv_RMSE"]),
                       f2(g_.loc["5-fold (shuffled)", "RMSE"]), f2(g_.loc["Forward chaining (5 folds)", "cv_RMSE"]),
                       f2(g_.loc["Forward chaining (5 folds)", "RMSE"])])
    write("Table4.csv", T4)
    F["tune_kfold_cv_min"], F["tune_kfold_cv_max"] = f1(kf.cv_RMSE.min()), f1(kf.cv_RMSE.max())
    F["tune_kfold_test_min"], F["tune_kfold_test_max"] = f1(kf.RMSE.min()), f1(kf.RMSE.max())
    F["tune_kfold_ratio_min"] = n(100 * (kf.cv_RMSE / kf.RMSE).min()); F["tune_kfold_ratio_max"] = n(100 * (kf.cv_RMSE / kf.RMSE).max())
    for d, tag in [("temporal_per_well", "perwell"), ("temporal_global", "global")]:
        rr = (fw[fw.design == d].cv_RMSE / fw[fw.design == d].RMSE - 1) * 100
        F[f"tune_fwd_{tag}_bias_min"] = n(rr.abs().min()); F[f"tune_fwd_{tag}_bias_max"] = n(rr.abs().max())
        F[f"tune_fwd_{tag}_bias_sign"] = "over" if (rr > 0).all() else ("under" if (rr < 0).all() else "mixed")
    dft = tu[tu.scheme == "Default"].set_index(["design", "model"]).RMSE
    ch = pd.concat([(g.set_index(["design", "model"]).RMSE - dft) for g in (kf, fw)])
    F["tune_ens_change_max_abs"] = f1(ch[ch.index.get_level_values("model") != "Decision Tree"].abs().max())
    F["tune_hours_total"] = f1(tu.search_seconds.sum() / 3600)
    for d in ["temporal_per_well", "temporal_global"]:
        for mdl in MODELS:
            for sch, sk in [("Default", "default"), ("5-fold (shuffled)", "kfold"), ("Forward chaining (5 folds)", "fwd")]:
                q = tu[(tu.design == d) & (tu.model == mdl) & (tu.scheme == sch)].iloc[0]
                F[f"tune_{d}_{SHORT[mdl]}_{sk}_RMSE"] = f2(q.RMSE)
    td = json.load(open(OUT / "tuning_design.json"))
    TG = [["Learner", "Hyperparameter", "Values searched"]]
    for mdl, grid in td["grids"].items():
        for k, v in grid.items(): TG.append([mdl, k.replace("m__", ""), ", ".join(v)])
    write("TableS_tuning_grids.csv", TG)
    TB = [["Design", "Block", "First month", "Last month"]]
    for d, blocks in td["forward_chaining_blocks"].items():
        for i, (a, b_) in enumerate(blocks): TB.append([DNAME[d], str(i + 1), a[:7], b_[:7]])
    write("TableS_tuning_blocks.csv", TB)
    PN = {"max_depth": "depth", "min_samples_split": "split", "min_samples_leaf": "leaf", "max_features": "features"}
    TSel = [["Design", "Learner", "Tuning", "Selected hyperparameters", "CV estimate of RMSE (m)", "Test RMSE (m)", "Test R²"]]
    for d in ["temporal_per_well", "temporal_global"]:
        first = True
        for mdl in MODELS:
            for sch, lab_ in [("5-fold (shuffled)", "Shuffled 5-fold"), ("Forward chaining (5 folds)", "Forward chaining")]:
                q = tu[(tu.design == d) & (tu.model == mdl) & (tu.scheme == sch)].iloc[0]
                bp = ", ".join(f"{PN.get(k.replace('m__', ''), k)} {'none' if v is None else v}" for k, v in json.loads(q.best_params).items())
                TSel.append([DNAME[d] if first else "", mdl, lab_, bp, f2(q.cv_RMSE), f2(q.RMSE), f2(q.R2)]); first = False
    write("TableS_tuning_selected.csv", TSel)

# ======================================================================== importance
imp = opt("importance_permutation.csv")
if imp is not None:
    for d in DESIGNS:
        gq = imp[(imp.design == d) & imp.is_group]
        for _, rr in gq.iterrows(): F[f"imp_{d}_{rr.unit.replace('group:', '')}"] = f2(rr.r2_drop_mean)
        F[f"imp_{d}_base_R2"] = f2(gq.base_R2.iloc[0])
    cm = imp[imp.is_group & (imp.unit == "group:climate_monthly")]
    F["imp_monthly_max"] = f2(cm.r2_drop_mean.max()); F["imp_monthly_min"] = f2(cm.r2_drop_mean.min())
    lcg = imp[imp.is_group & (imp.unit == "group:land_cover")]; assert len(lcg) == len(DESIGNS)
    F["imp_landcover_max_abs"] = f3(lcg.r2_drop_mean.abs().max())

# ======================================================================== spatial error by latitude band
sb = opt("spatial_bands.csv")
if sb is not None:
    BANDS = [("(-90.0, -36.0]", "s36", "south of 36° S"), ("(-36.0, -30.0]", "30to36", "30–36° S"),
             ("(-30.0, -26.0]", "26to30", "26–30° S"), ("(-26.0, 0.0]", "n26", "north of 26° S")]
    TSB = [["Latitude band of test wells", "Seeds with test wells", "Test wells (mean)", "Share of test well-months (%)",
            "Training mean RMSE (m)", "Neighbor interpolation RMSE (m)", "Extra Trees RMSE (m)", "Extra Trees skill (seeds positive)",
            "Random Forest skill (seeds positive)", "Decision Tree skill (seeds positive)", "Share of Extra Trees squared error (%)"]]
    seeds_all = sorted(sb.seed.unique())   # shares are averaged over all seeds, with 0 where a seed has no test wells in the band
    tot_se, tot_n = {}, {}
    for seed, g in sb[sb.model == "Extra Trees"].groupby("seed"): tot_se[seed] = ((g.RMSE ** 2) * g.n_test).sum(); tot_n[seed] = g.n_test.sum()
    for band, tag, lab_ in BANDS:
        q = sb[sb.band == band]
        if q.empty: continue
        et = q[q.model == "Extra Trees"].set_index("seed"); tm = q[q.model == "Training mean"].set_index("seed")
        nb = q[q.model == "Neighbor interpolation (k=5, IDW)"].set_index("seed")
        ss = 1 - (et.RMSE / tm.RMSE.reindex(et.index)) ** 2
        share = 100 * pd.Series({s_: (et.RMSE[s_] ** 2 * et.n_test[s_]) / tot_se[s_] for s_ in et.index}).reindex(seeds_all, fill_value=0.0)
        nshare = 100 * pd.Series({s_: et.n_test[s_] / tot_n[s_] for s_ in et.index}).reindex(seeds_all, fill_value=0.0)
        F[f"sb_{tag}_skill"] = f2(ss.mean()); F[f"sb_{tag}_npos"] = str(int((ss > 0).sum())); F[f"sb_{tag}_nseeds"] = str(len(ss))
        F[f"sb_{tag}_share_mean"] = f1(share.mean()); F[f"sb_{tag}_testshare_mean"] = f1(nshare.mean())
        F[f"sb_{tag}_et_RMSE"] = f1(et.RMSE.mean()); F[f"sb_{tag}_mean_RMSE"] = f1(tm.RMSE.mean())
        # skill pooled over seeds (sum of squared errors over all seeds' test well-months in the band)
        F[f"sb_{tag}_skill_pooled"] = f2(1 - (et.RMSE ** 2 * et.n_test).sum() / (tm.RMSE.reindex(et.index) ** 2 * et.n_test).sum())
        cell = {}
        for mdl in ["Decision Tree", "Random Forest", "Extra Trees"]:
            o = q[q.model == mdl].set_index("seed"); so = 1 - (o.RMSE / tm.RMSE.reindex(o.index)) ** 2
            F[f"sb_{tag}_{SHORT[mdl]}_skill"] = f2(so.mean()); F[f"sb_{tag}_{SHORT[mdl]}_npos"] = str(int((so > 0).sum()))
            F[f"sb_{tag}_{SHORT[mdl]}_nneg"] = str(int((so <= 0).sum()))
            cell[mdl] = f2(so.mean()) + (" ± " + f2(so.std(ddof=1)) if len(so) > 1 else "") + f" ({int((so > 0).sum())} of {len(so)})"
        TSB.append([lab_, str(len(ss)), f1(et.n_wells.mean()), f1(nshare.mean()), f2(tm.RMSE.mean()), f2(nb.RMSE.mean()), f2(et.RMSE.mean()),
                    cell["Extra Trees"], cell["Random Forest"], cell["Decision Tree"], f1(share.mean())])
    write("TableS_spatial_bands.csv", TSB)

# ======================================================================== chronological diagnostics (per-well design, seed 42)
trp, tep, _ = rp.DESIGNS["temporal_per_well"](t, None)
pc = pd.read_parquet(OUT / "predictions" / "main_temporal_per_well_seed42.parquet"); assert np.array_equal(pc.row.to_numpy(), tep)
for b_, v in rp.baseline_predictions(t, trp, tep, "temporal_per_well").items(): pc[b_] = v
for mdl in MODELS + ["Last observation (persistence)", "Well mean (training records)"]:
    F[f"good_{KEY[mdl]}_pct"] = f1(100 * ((pc[mdl] - pc.y_true).abs() <= 5).mean())
F["good_threshold_m"] = "5"
poor = (pc["Extra Trees"] - pc.y_true).abs() > 5
for lo, hi, tag in [(0, 5, "lt5"), (5, 20, "5to20"), (20, 50, "20to50"), (50, 1e9, "gt50")]:
    sel = (pc.y_true > lo) & (pc.y_true <= hi); F[f"poor_{tag}_pct"] = f1(100 * poor[sel].mean())
F["temporal_per_well_et_bias_all"] = f1((pc["Extra Trees"] - pc.y_true).mean())
F["temporal_per_well_last_bias_all_abs"] = f1(abs((pc["Last observation (persistence)"] - pc.y_true).mean()))
F["temporal_per_well_last_bias_all"] = f1((pc["Last observation (persistence)"] - pc.y_true).mean())
F["temporal_per_well_wellmean_bias_all"] = f1((pc["Well mean (training records)"] - pc.y_true).mean())
# is the test-period level deeper than the last training value? (both chronological designs)
for d in ["temporal_per_well", "temporal_global"]:
    trd_, ted_, _ = rp.DESIGNS[d](t, None)
    q_ = t.iloc[ted_][[WELL, TARGET]].copy(); q_["last"] = rp.baseline_predictions(t, trd_, ted_, d)["Last observation (persistence)"]
    w_ = q_.groupby(WELL).agg(te=(TARGET, "mean"), last=("last", "first"))
    F[f"{d}_pct_wells_deeper_than_last"] = f1(100 * (w_.te > w_["last"]).mean()); F[f"{d}_median_deeper_than_last"] = f2((w_.te - w_["last"]).median())
    F[f"{d}_n_test_wells"] = n(len(w_)); F[f"{d}_last_bias_all"] = f1((q_["last"] - q_[TARGET]).mean())
se = pc.assign(a=(pc["Extra Trees"] - pc.y_true) ** 2, b=(pc["Last observation (persistence)"] - pc.y_true) ** 2,
               dev=(pc.y_true - pc["Well mean (training records)"]).abs())
pw = se.groupby("well").agg(mse=("a", "mean"), msep=("b", "mean"), dev=("dev", "mean"), n_te=("a", "size"))
pw["pos"] = (1 - pw.mse / pw.msep) > 0; pw["rmse"] = pw.mse ** 0.5
pw = pw.join(t.groupby(WELL)[["Basin", "well_lat"]].first())
F["pw_wells"] = n(len(pw)); F["pw_wells_pos"] = n(pw.pos.sum()); F["pw_wells_pos_pct"] = f1(100 * pw.pos.mean())
for tag, sel in [("s36", pw.well_lat < -36), ("n26", pw.well_lat > -26), ("26to36", pw.well_lat.between(-36, -26))]:
    F[f"pw_pos_pct_{tag}"] = f1(100 * pw[sel].pos.mean()); F[f"pw_wells_{tag}"] = n(sel.sum())
F["pw_dev_q25_pos"], F["pw_dev_q75_pos"] = f1(pw[pw.pos].dev.quantile(.25)), f1(pw[pw.pos].dev.quantile(.75))
F["pw_dev_q25_neg"], F["pw_dev_q75_neg"] = f1(pw[~pw.pos].dev.quantile(.25)), f1(pw[~pw.pos].dev.quantile(.75))
BASIN_PRETTY = {"COSTERAS R.ELQUI-R.LIMARI": "coastal basins between the Elqui and Limarí rivers", "RIO COPIAPO": "Río Copiapó basin",
                "COSTERAS ACONCAGUA-MAIPO": "coastal basins between the Aconcagua and Maipo rivers", "RIO MAIPO": "Río Maipo basin",
                "RIO LIGUA": "Río La Ligua basin", "RIO ELQUI": "Río Elqui basin", "RIO RAPEL": "Río Rapel basin", "RIO ACONCAGUA": "Río Aconcagua basin"}
bb = pw.groupby("Basin").rmse.agg(["median", "size"]).query("size >= 15").sort_values("median", ascending=False)
for i in range(3):
    F[f"rmse_basin{i+1}_name"] = BASIN_PRETTY.get(bb.index[i], bb.index[i].title()); F[f"rmse_basin{i+1}_median"] = f1(bb["median"].iloc[i])
F["rmse_basins_min_wells"] = "15"
for d in ["temporal_per_well", "temporal_global"]:
    pp = pd.read_parquet(OUT / "predictions" / f"main_{d}_seed42.parquet")
    is_te = np.isin(np.arange(len(t)), pp.row.to_numpy())
    tr_ = t[~is_te & t[WELL].isin(set(pp.well))].groupby(WELL)[TARGET].mean(); te_ = t[is_te].groupby(WELL)[TARGET].mean()
    dlt = (te_ - tr_.reindex(te_.index)).dropna()
    F[f"{d}_pct_wells_deeper"] = f1(100 * (dlt > 0).mean()); F[f"{d}_median_deepening"] = f1(dlt.median())
for d in DESIGNS:
    pp = pd.read_parquet(OUT / "predictions" / f"main_{d}_seed42.parquet"); e = pp["Extra Trees"] - pp.y_true
    for tag, sel in [("shallow", pp.y_true <= 5), ("deep", pp.y_true > 50)]:
        F[f"{d}_bias_{tag}_abs"] = f1(abs(e[sel].mean())); F[f"{d}_bias_{tag}_sign"] = "over" if e[sel].mean() > 0 else "under"
f8 = FIG / "Figure8_wells.json"
if f8.exists():
    w8 = json.load(open(f8))
    F["hydro_rmse_min"] = f1(min(x["rmse_extra_trees"] for x in w8["wells"])); F["hydro_rmse_max"] = f1(max(x["rmse_extra_trees"] for x in w8["wells"]))
    if "eligible_wells" in w8: F["hydro_eligible_wells"] = n(w8["eligible_wells"]); F["hydro_eligible_pos_pct"] = f1(100 * w8["eligible_wells_extra_trees_better"] / w8["eligible_wells"])

# ======================================================================== supplementary predictor dictionary
BIO = {1: ("Annual mean temperature", "°C × 10"), 2: ("Mean diurnal range", "°C × 10"), 3: ("Isothermality (BIO2/BIO7 × 100)", "%"),
       4: ("Temperature seasonality (standard deviation of monthly mean temperature)", "°C × 1000"), 5: ("Maximum temperature of warmest month", "°C × 10"),
       6: ("Minimum temperature of coldest month", "°C × 10"), 7: ("Temperature annual range (BIO5 − BIO6)", "°C × 10"),
       8: ("Mean temperature of wettest quarter", "°C × 10"), 9: ("Mean temperature of driest quarter", "°C × 10"),
       10: ("Mean temperature of warmest quarter", "°C × 10"), 11: ("Mean temperature of coldest quarter", "°C × 10"),
       12: ("Annual precipitation", "mm"), 13: ("Precipitation of wettest month", "mm"), 14: ("Precipitation of driest month", "mm"),
       15: ("Precipitation seasonality (coefficient of variation)", "%"), 16: ("Precipitation of wettest quarter", "mm"),
       17: ("Precipitation of driest quarter", "mm"), 18: ("Precipitation of warmest quarter", "mm"), 19: ("Precipitation of coldest quarter", "mm")}
TC = {"pr": ("Precipitation", "mm"), "tmmn": ("Minimum temperature", "°C"), "tmmx": ("Maximum temperature", "°C"),
      "aet": ("Actual evapotranspiration", "mm"), "pet": ("Reference evapotranspiration", "mm"), "def": ("Climatic water deficit", "mm"),
      "pdsi": ("Palmer Drought Severity Index", "unitless"), "srad": ("Downward surface shortwave radiation", "W m⁻²"),
      "vpd": ("Vapor pressure deficit", "kPa"), "vs": ("Wind speed at 10 m", "m s⁻¹")}
S1 = [["Predictor (column)", "Family", "Description", "Units", "Product (identifier)", "Time stamp"],
      ["cop_dem_30_DEM_value", "Topography", "Elevation (digital surface model)", "m", "Copernicus DEM GLO-30 (COPERNICUS/DEM/GLO30)", "Static"],
      ["elevation_NASADEM", "Topography", "Elevation", "m", "NASADEM (NASA/NASADEM_HGT/001)", "Static"],
      ["slope_NASADEM", "Topography", "Slope derived from NASADEM", "degrees", "NASADEM (NASA/NASADEM_HGT/001)", "Static"],
      ["aspect_NASADEM", "Topography", "Aspect derived from NASADEM", "degrees from north", "NASADEM (NASA/NASADEM_HGT/001)", "Static"],
      ["elevation_Alos_Palsar", "Topography", "Elevation (digital surface model band; from the ALOS PRISM sensor, despite the column name)", "m", "AW3D30 v3.2 (JAXA/ALOS/AW3D30/V3_2)", "Static"]]
S1 += [[f"wclim_bio_bio{i:02d}_value", "Long-term climate", f"BIO{i}: {BIO[i][0]}", BIO[i][1], "WorldClim 1.4 (WORLDCLIM/V1/BIO)", "Static (long-term normals)"] for i in range(1, 20)]
TC_SERVER = {"pr": "ppt", "tmmn": "tmin", "tmmx": "tmax", "vs": "ws", "pdsi": "PDSI"}   # server variable names (fetch_terraclimate_bbox.py)
S1 += [[f"tc_{b_}", "Monthly climate", TC[b_][0], TC[b_][1], f"TerraClimate (server variable {TC_SERVER.get(b_, b_)})", "Record's calendar month"] for b_ in TC]
S1 += [["land_cover_igbp", "Land cover", "IGBP land-cover class, most frequent class over the well's records", "class code", "MODIS MCD12Q1 v061 (MODIS/061/MCD12Q1)", "Static per well"],
       ["Longitude_GCS_WGS_1984", "Location", "Well longitude", "decimal degrees (WGS 84)", "Source compilation", "Static"],
       ["Latitude_GCS_WGS_1984", "Location", "Well latitude", "decimal degrees (WGS 84)", "Source compilation", "Static"]]
assert [r[0] for r in S1[1:]] == feats, "the dictionary must list exactly the pipeline's predictors, in order"
write("TableS_dictionary.csv", S1)
F["excluded_fields"] = man["design"]["excluded"].replace(" (features)", "")
import datetime as _dt   # access date of the TerraClimate re-extraction = time stamp of its log
F["tc_access_date"] = _dt.datetime.strptime(tcv["accessed"], "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")
F["tc_file_pattern"] = "agg_terraclimate_<variable>_1950_CurrentYear_GLOBE.nc"
dep = HERE / "deposit.json"
F["zenodo_doi_url"] = json.load(open(dep))["zenodo_doi_url"] if dep.exists() else "https://doi.org/10.5281/zenodo.XXXXXXX"

json.dump(F, open(OUT / "facts.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(F), "facts;", "tables:", sorted(p.name for p in TAB.glob("*.csv")))
