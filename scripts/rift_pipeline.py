"""
RIFT clean re-analysis (v2.1, after independent code review).

One pipeline, one dataset, one declared predictor set. Every number reported in the manuscript must come from the
files this script writes under OUT. Stages cache their outputs; rerun a stage with --force.

    python rift_pipeline.py --stage all
    python rift_pipeline.py --stage main --force

Changes after review (see pipeline_review_result.json):
  * Physical-well key (Name + ~1 km location, renamed wells merged via shared Code); co-location groups (< 100 m)
    are kept together in the well-based split.
  * TerraClimate re-extracted and joined on each record's OWN calendar month (the GEE extraction rounded dates, giving
    54% of records the next month's climate). MOD16 excluded (dated 2025-06-10 for ~96% of records).
  * Land cover fixed per well (mode of MODIS IGBP class), not averaged and not time-varying.
  * Robust error measures (WAPE, MdAPE, MAPE for depth >= 1 m) alongside MAPE.
  * Well-mean baseline for every design that keeps wells in both partitions; kNN baseline on one row per well.
  * Temporal designs repeat the model over the 5 seeds (fixed split) so they carry model-seed variability.
  * Sensitivity and tuning also run on the causal global-origin split; min-records sensitivity subsets wells first.
  * Forward-chaining folds are built on whole calendar months. Grouped permutation importance on held-out data.
  * v2.2: admissible baselines are also written for the sensitivity partitions (stage 'sensbase'), and
    error by horizon for learners and baselines on the chronological partitions (stage 'horizon').
  * v2.3 (audit): neighbor-interpolation baseline on geographic distance from OTHER wells (the v2.2 kNN used
    standardized coordinates and returned the test well itself under the random split); temporal-interpolation
    baseline for the random split; drop-column ablations without monthly climate and with longitude only;
    spatial-block error by latitude band.
"""
import argparse, hashlib, json, platform, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
CSV = Path(os.environ.get("RIFT_SOURCE_CSV", HERE.parent / "data" / "raw" /
                          "groundwater_chile_and_elevation_dataset_2025_with_GEE_7_2_25.csv"))  # source file from the Zenodo deposit
OUT = HERE / "outputs"
TC_FILE = OUT / "terraclimate_by_location_month.parquet"
TARGET = "Depth to water (m)"
WELL = "well_id"
LON, LAT = "Longitude_GCS_WGS_1984", "Latitude_GCS_WGS_1984"
SEEDS = [42, 43, 44, 45, 46]
TEST_FRAC = 0.30
GRID_N = 10
MIN_RECORDS_TEMPORAL = 3
N_JOBS = 6
TC_BANDS = ["pr", "tmmn", "tmmx", "aet", "pet", "def", "pdsi", "srad", "vpd", "vs"]

PREDICTORS = {
    "topography": ["cop_dem_30_DEM_value", "elevation_NASADEM", "slope_NASADEM", "aspect_NASADEM", "elevation_Alos_Palsar"],
    "climate_longterm": [f"wclim_bio_bio{i:02d}_value" for i in range(1, 20)],
    "climate_monthly": [f"tc_{b}" for b in TC_BANDS],
    "land_cover": ["land_cover_igbp"],
    "location": [LON, LAT],
}
FEATURES = [c for grp in PREDICTORS.values() for c in grp]
FEATURES_NOCOORD = [c for c in FEATURES if c not in (LON, LAT)]
FEATURES_NOMONTHLY = [c for c in FEATURES if c not in PREDICTORS["climate_monthly"]]
FEATURES_LONONLY = [LON]
ABLATIONS = {"nocoord": FEATURES_NOCOORD, "nomonthly": FEATURES_NOMONTHLY, "lononly": FEATURES_LONONLY}
KNN_K = 5
LAT_BANDS = [-90.0, -36.0, -30.0, -26.0, 0.0]
STATIC_RAW = PREDICTORS["topography"] + PREDICTORS["climate_longterm"]

DESIGN = {
    "record_filter": "Status blank (static-level readings); target not null; depth >= 0; dataset Outlier flag False",
    "well_key": "Name + location rounded to 0.01 deg (~1 km); keys sharing a non-null Code within 1 km are merged "
                "(renamed wells); co-location groups = wells within 100 m, kept together in the well-based split",
    "aggregation": "mean per (well, calendar month); climate joined per record on its own calendar month before averaging",
    "climate": "TerraClimate re-extracted from the public THREDDS server (NCSS), nearest 1/24 deg cell, converted to "
               "physical units with each file's scale factor and offset",
    "land_cover": "per-well mode of MODIS MCD12Q1 IGBP class over the well's records (static)",
    "excluded": "predictors are an explicit include-list (features); every other source column is dropped, including "
                "row IDs, well names and codes, dates and timestamps, *_rs_date, *_asset_id, *_unit, *_description, *_scale, "
                "*_normalize, *_pixel_size, *_processing_method, *_status, original coordinates, coordinate system, EPSG, "
                "Basin, Sub_Basin, Status and Outlier (used only as filters), well-metadata Elevation, the compilation's GEE "
                "TerraClimate values (replaced by the re-extraction), MODIS land-cover types 2-5 and MOD16A2",
    "models": "DecisionTree, RandomForest(100), ExtraTrees(100); scikit-learn defaults otherwise",
    "splits": {
        "random": "70/30 of well-month records; 5 seeds (split and model)",
        "well": "GroupShuffleSplit on co-location groups, 30% held out; 5 seeds",
        "spatial": "10x10 equal-angle grid over the well bounding box; whole occupied cells drawn in shuffled order, a cell "
                   "that would overshoot 30% is skipped if stopping is closer to 30%; no buffer; 5 seeds",
        "temporal_per_well": "per well sorted by month, first floor(0.7 n) well-months train, rest test; wells with <= 3 "
                             "well-months train only; fixed split, model seeds 42-46",
        "temporal_global": "one calendar origin at the 70th percentile of record months; test = records on/after origin at "
                           "wells with training history; fixed split, model seeds 42-46",
    },
    "baselines": "training mean; well training mean (designs sharing wells); last observation (temporal); neighbor "
                 "interpolation (inverse-distance mean of the training means of the 5 geographically nearest other wells, "
                 "haversine; non-temporal designs); temporal interpolation (linear in time between the same well's "
                 "training well-months, constant beyond the ends; random design)",
    "ablations": "nocoord: without the two coordinates; nomonthly: without the 10 monthly climate variables; "
                 "lononly: longitude only",
}

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); e = np.abs(p - y)
    ape = e / np.abs(y)
    ge1 = y >= 1.0
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": mean_absolute_error(y, p),
            "MAPE": float(100 * ape.mean()), "MAPE_ge1m": float(100 * ape[ge1].mean()) if ge1.any() else np.nan,
            "MdAPE": float(100 * np.median(ape)), "WAPE": float(100 * e.sum() / np.abs(y).sum()),
            "n_test": int(len(y)), "n_test_lt1m": int((~ge1).sum())}

def tree_models(seed):
    return {"Decision Tree": DecisionTreeRegressor(random_state=seed),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=N_JOBS),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=N_JOBS)}

def fit_predict(model, Xtr, ytr, Xte):
    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), clone(model))
    t = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t
    return pipe, pipe.predict(Xte), fit_s

def check_index(t):
    assert t.index.equals(pd.RangeIndex(len(t))), "model table must carry a RangeIndex"

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

class UF:
    def __init__(self, items): self.p = {i: i for i in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)

# ------------------------------------------------------------------ data
def build_well_keys(d):
    """Physical wells: Name@location(0.01 deg); merge keys sharing a Code if within 1 km."""
    d = d.copy()
    d["node"] = d["Name"].str.strip() + "@" + d[LON].round(2).astype(str) + "," + d[LAT].round(2).astype(str)
    nodes = d.groupby("node").agg(lon=(LON, "median"), lat=(LAT, "median"))
    uf = UF(nodes.index)
    merged = 0
    for code, g in d.dropna(subset=["Code"]).groupby("Code"):
        ns = g["node"].unique()
        for a in ns[1:]:
            if haversine_km(nodes.lon[ns[0]], nodes.lat[ns[0]], nodes.lon[a], nodes.lat[a]) <= 1.0:
                uf.union(a, ns[0]); merged += 1
    d[WELL] = d["node"].map(lambda n: uf.find(n))
    wl = d.groupby(WELL).agg(well_lon=(LON, "median"), well_lat=(LAT, "median"))
    # co-location groups: wells within 100 m
    uf2 = UF(wl.index); ids = wl.index.to_numpy(); lo = wl.well_lon.to_numpy(); la = wl.well_lat.to_numpy()
    for i in range(len(ids)):
        dist = haversine_km(lo[i], la[i], lo[i + 1:], la[i + 1:])
        for j in np.where(dist <= 0.1)[0]:
            uf2.union(ids[i + 1 + j], ids[i])
    wl["coloc_group"] = [uf2.find(i) for i in wl.index]
    spread = d.groupby(WELL).apply(lambda g: haversine_km(g[LON].min(), g[LAT].min(), g[LON].max(), g[LAT].max()),
                                   include_groups=False)
    return d, wl, {"code_merges": merged, "max_within_well_spread_km": float(spread.max()),
                   "colocation_groups": int(wl.coloc_group.nunique())}

def stage_prepare(force=False):
    f = OUT / "model_table.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    raw = pd.read_csv(CSV, encoding="latin-1", low_memory=False)
    sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
    counts = {"raw_rows": len(raw)}
    d = raw[raw["Status"].isna()]; counts["status_blank"] = len(d)
    d = d[d[TARGET].notna()]; counts["target_present"] = len(d)
    d = d[d[TARGET] >= 0]; counts["non_negative"] = len(d)
    d = d[~d["Outlier"].astype(bool)]; counts["outlier_flag_false"] = len(d)
    d = d.assign(month=pd.to_datetime(d["Date_String"]).dt.to_period("M").dt.to_timestamp())
    d, wl, keyinfo = build_well_keys(d)
    assert keyinfo["max_within_well_spread_km"] < 1.5, keyinfo
    # climate on the record's own calendar month
    tc = pd.read_parquet(TC_FILE).rename(columns={b: f"tc_{b}" for b in TC_BANDS})
    d = d.assign(lon5=d[LON].round(5), lat5=d[LAT].round(5)).merge(
        tc.rename(columns={"lon": "lon5", "lat": "lat5"}), on=["lon5", "lat5", "month"], how="left")
    miss = d[[f"tc_{b}" for b in TC_BANDS]].isna().any(axis=1)
    counts["climate_missing_rows_dropped"] = int(miss.sum()); d = d[~miss]
    # static land cover per well
    lc = d.groupby(WELL)["modis_lc_LC_Type1_value"].agg(lambda s: s.mode().iloc[0])
    d["land_cover_igbp"] = d[WELL].map(lc)
    agg = {TARGET: "mean", "Basin": "first", "Name": "first",
           **{c: "mean" for c in FEATURES if c not in ("land_cover_igbp",)}, "land_cover_igbp": "first"}
    t = d.groupby([WELL, "month"], as_index=False).agg(agg)
    counts["well_months"] = len(t)
    counts["well_months_from_multiple_records"] = int((d.groupby([WELL, "month"]).size() > 1).sum())
    t = t.merge(wl, left_on=WELL, right_index=True).sort_values([WELL, "month"]).reset_index(drop=True)
    t["Basin"] = t["Basin"].astype(str).str.strip()
    counts["wells"] = int(t[WELL].nunique()); counts.update(keyinfo)
    counts["date_first"] = str(t.month.min().date()); counts["date_last"] = str(t.month.max().date())
    t.to_parquet(f)
    json.dump({"csv": str(CSV), "sha256": sha, "counts": counts}, open(OUT / "data_provenance.json", "w"), indent=1)
    log(f"prepare: {counts}")
    return t

# ------------------------------------------------------------------ splits (all return positional indices)
def split_random(t, seed):
    tr, te = train_test_split(np.arange(len(t)), test_size=TEST_FRAC, random_state=seed)
    return np.sort(tr), np.sort(te), {}

def split_well(t, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr, te = next(gss.split(t, groups=t["coloc_group"]))
    return tr, te, {}

def grid_cells(t):
    wl = t.groupby(WELL)[["well_lon", "well_lat"]].first()
    lon_edges = np.linspace(wl.well_lon.min(), wl.well_lon.max(), GRID_N + 1)
    lat_edges = np.linspace(wl.well_lat.min(), wl.well_lat.max(), GRID_N + 1)
    ci = np.clip(np.digitize(wl.well_lon, lon_edges) - 1, 0, GRID_N - 1)
    cj = np.clip(np.digitize(wl.well_lat, lat_edges) - 1, 0, GRID_N - 1)
    return pd.Series(ci * GRID_N + cj, index=wl.index), lon_edges, lat_edges

def split_spatial(t, seed):
    cell_of_well, lon_e, lat_e = grid_cells(t)
    cell = t[WELL].map(cell_of_well).to_numpy()
    counts = pd.Series(cell).value_counts()
    cells = counts.index.to_numpy().copy(); np.random.default_rng(seed).shuffle(cells)
    target = TEST_FRAC * len(t); chosen, n = [], 0
    for c in cells:
        if n >= target: break
        if n + counts[c] > target and abs(n - target) < abs(n + counts[c] - target):
            continue                      # skip a cell that overshoots when stopping is closer to 30%
        chosen.append(c); n += counts[c]
    te_mask = np.isin(cell, chosen)
    tr, te = np.where(~te_mask)[0], np.where(te_mask)[0]
    wl = t.groupby(WELL)[["well_lon", "well_lat"]].first()
    tw, rw = wl.loc[t.loc[te, WELL].unique()], wl.loc[t.loc[tr, WELL].unique()]
    dmin = np.array([haversine_km(x, y, rw.well_lon.values, rw.well_lat.values).min() for x, y in zip(tw.well_lon, tw.well_lat)])
    return tr, te, {"test_cells": len(chosen), "occupied_cells": len(counts),
                    "cell_deg_lon": float(lon_e[1] - lon_e[0]), "cell_deg_lat": float(lat_e[1] - lat_e[0]),
                    "test_to_nearest_train_km_median": float(np.median(dmin)), "test_to_nearest_train_km_min": float(dmin.min())}

def split_temporal_per_well(t, seed=None, frac_train=1 - TEST_FRAC, min_records=MIN_RECORDS_TEMPORAL):
    tr, te, horizon = [], [], []
    pos = pd.Series(np.arange(len(t)), index=t.index)
    for _, g in t.groupby(WELL, sort=False):
        g = g.sort_values("month"); idx = pos[g.index].to_numpy(); n = len(g)
        if n <= min_records:
            tr.extend(idx); continue
        k = int(np.floor(frac_train * n + 1e-9)); k = min(max(k, 1), n - 1)
        tr.extend(idx[:k]); te.extend(idx[k:])
        origin = g["month"].iloc[k - 1]
        horizon.extend(((g["month"].iloc[k:] - origin).dt.days / 365.25).tolist())
    tr, te, h = np.array(tr), np.array(te), np.array(horizon)
    return tr, te, {"horizon_median_y": float(np.median(h)), "horizon_q25_y": float(np.percentile(h, 25)),
                    "horizon_q75_y": float(np.percentile(h, 75)),
                    "test_before_last_train_month_pct": float(100 * np.mean(t["month"].to_numpy()[te] < t["month"].to_numpy()[tr].max()))}

def split_temporal_global(t, seed=None, pct=70):
    origin = pd.Timestamp(np.percentile(t["month"].astype("int64"), pct)).to_period("M").to_timestamp()
    tr_mask = (t["month"] < origin).to_numpy()
    trained = set(t.loc[tr_mask, WELL])
    te_mask = (t["month"] >= origin).to_numpy() & t[WELL].isin(trained).to_numpy()
    last_train = origin - pd.DateOffset(months=1)                 # horizons measured from the last training month
    h = (t.loc[te_mask, "month"] - last_train).dt.days / 365.25
    return np.where(tr_mask)[0], np.where(te_mask)[0], {
        "origin": str(origin.date()), "last_train_month": str(last_train.date()), "horizon_median_y": float(h.median()),
        "horizon_q25_y": float(h.quantile(.25)), "horizon_q75_y": float(h.quantile(.75))}

DESIGNS = {"random": split_random, "well": split_well, "spatial": split_spatial,
           "temporal_per_well": split_temporal_per_well, "temporal_global": split_temporal_global}
FIXED_SPLIT = {"temporal_per_well", "temporal_global"}

# ------------------------------------------------------------------ baselines
def baselines(t, tr, te, design):
    yte = t[TARGET].to_numpy()[te]
    return {k: metrics(yte, v) for k, v in baseline_predictions(t, tr, te, design).items()}

def neighbor_interpolation(trd, ted, k=KNN_K):
    """Inverse-distance mean of the training means of the k geographically nearest OTHER wells (haversine distance)."""
    from sklearn.neighbors import BallTree
    wl = trd.groupby(WELL).agg(lat=("well_lat", "first"), lon=("well_lon", "first"), v=(TARGET, "mean"))
    tree = BallTree(np.radians(wl[["lat", "lon"]].to_numpy()), metric="haversine")
    tw = ted.groupby(WELL).agg(lat=("well_lat", "first"), lon=("well_lon", "first"))
    dist, idx = tree.query(np.radians(tw[["lat", "lon"]].to_numpy()), k=min(k + 1, len(wl)))
    dist = dist * 6371.0; ids = wl.index.to_numpy(); vals = wl.v.to_numpy(); pred = {}
    for j, w in enumerate(tw.index):
        keep = ids[idx[j]] != w                                  # never use the test well's own training record
        dd, vv = dist[j][keep][:k], vals[idx[j]][keep][:k]
        wgt = 1.0 / np.maximum(dd, 0.01)                         # 10 m floor for co-located wells
        pred[w] = float((wgt * vv).sum() / wgt.sum())
    return ted[WELL].map(pred).to_numpy()

def temporal_interpolation(trd, ted, fallback):
    """Linear interpolation in time between the same well's training well-months, constant beyond the ends."""
    out = np.full(len(ted), fallback, float)
    tm = (ted["month"].dt.year * 12 + ted["month"].dt.month).to_numpy()
    pos = pd.Series(np.arange(len(ted)), index=ted.index)
    grp = trd.groupby(WELL)
    for w, g in ted.groupby(WELL):
        if w not in grp.groups: continue
        h = grp.get_group(w).sort_values("month")
        x = (h["month"].dt.year * 12 + h["month"].dt.month).to_numpy()
        ii = pos[g.index].to_numpy(); out[ii] = np.interp(tm[ii], x, h[TARGET].to_numpy())
    return out

def baseline_predictions(t, tr, te, design):
    ytr = t[TARGET].to_numpy()[tr]
    trd, ted = t.iloc[tr], t.iloc[te]
    out = {"Training mean": np.full(len(te), ytr.mean())}
    wm = trd.groupby(WELL)[TARGET].mean()
    shared = ted[WELL].isin(wm.index).mean()
    if shared > 0.5:
        out["Well mean (training records)"] = ted[WELL].map(wm).fillna(ytr.mean()).to_numpy()
    if design.startswith("temporal"):
        last = trd.sort_values("month").groupby(WELL)[TARGET].last()
        out["Last observation (persistence)"] = ted[WELL].map(last).fillna(ytr.mean()).to_numpy()
    else:
        out["Neighbor interpolation (k=5, IDW)"] = neighbor_interpolation(trd, ted)
        if design == "random":
            out["Temporal interpolation (same well)"] = temporal_interpolation(trd, ted, ytr.mean())
    return out

# ------------------------------------------------------------------ stages
def stage_main(t, force=False, features=FEATURES, tag="main"):
    f = OUT / f"metrics_{tag}.csv"
    if f.exists() and not force:
        return pd.read_csv(f)
    check_index(t)
    rows, split_rows, assign = [], [], []
    (OUT / "predictions").mkdir(exist_ok=True)
    for design, fn in DESIGNS.items():
        for seed in SEEDS:
            if design in FIXED_SPLIT:
                tr, te, info = fn(t, None)
            else:
                tr, te, info = fn(t, seed)
            assert len(np.intersect1d(tr, te)) == 0
            if tag == "main":
                role = np.full(len(t), "excluded", dtype=object); role[tr] = "train"; role[te] = "test"
                assign.append(pd.DataFrame({"design": design, "seed": seed, "row": np.arange(len(t)), "role": role}))
            split_rows.append({"design": design, "seed": seed, "n_train": len(tr), "n_test": len(te),
                               "n_excluded": len(t) - len(tr) - len(te), "pct_train_of_table": 100 * len(tr) / len(t),
                               "pct_test_of_table": 100 * len(te) / len(t),
                               "wells_train": t.iloc[tr][WELL].nunique(), "wells_test": t.iloc[te][WELL].nunique(),
                               "test_wells_also_in_train": len(set(t.iloc[te][WELL]) & set(t.iloc[tr][WELL])), **info})
            Xtr, Xte = t.iloc[tr][features], t.iloc[te][features]
            ytr, yte = t[TARGET].to_numpy()[tr], t[TARGET].to_numpy()[te]
            preds = pd.DataFrame({"row": te, "well": t.iloc[te][WELL].to_numpy(), "month": t.iloc[te]["month"].to_numpy(), "y_true": yte})
            for name, model in tree_models(seed).items():
                pipe, p, fit_s = fit_predict(model, Xtr, ytr, Xte)
                trm = metrics(ytr, pipe.predict(Xtr))
                rows.append({"design": design, "seed": seed, "model": name, "fit_seconds": fit_s, **metrics(yte, p),
                             "train_R2": trm["R2"], "train_RMSE": trm["RMSE"]})
                preds[name] = p
                log(f"{tag} {design} seed={seed} {name}: R2={rows[-1]['R2']:.3f} RMSE={rows[-1]['RMSE']:.2f} WAPE={rows[-1]['WAPE']:.1f} ({fit_s:.0f}s)")
            for bname, m in baselines(t, tr, te, design).items():
                rows.append({"design": design, "seed": seed, "model": bname, "fit_seconds": 0.0, **m})
            preds.to_parquet(OUT / "predictions" / f"{tag}_{design}_seed{seed}.parquet")
            pd.DataFrame(rows).to_csv(f.with_suffix(".partial.csv"), index=False)
    pd.DataFrame(split_rows).to_csv(OUT / f"splits_{tag}.csv", index=False)
    if assign:
        a_ = pd.concat(assign, ignore_index=True)
        for col in ("design", "role"): a_[col] = a_[col].astype("category")
        a_.to_parquet(OUT / "split_assignments_main.parquet")    # row = position in model_table.parquet
    m = pd.DataFrame(rows); m.to_csv(f, index=False); return m

def stage_importance(t, force=False):
    """Grouped and per-feature permutation importance on held-out data (Random Forest, seed 42)."""
    f = OUT / "importance_permutation.csv"
    if f.exists() and not force: return
    rng = np.random.default_rng(42); rows = []
    for design, fn in DESIGNS.items():
        tr, te, _ = fn(t, None) if design in FIXED_SPLIT else fn(t, 42)
        pipe, _, _ = fit_predict(tree_models(42)["Random Forest"], t.iloc[tr][FEATURES], t[TARGET].to_numpy()[tr], t.iloc[te][FEATURES])
        sub = te if len(te) <= 5000 else rng.choice(te, 5000, replace=False)
        X = t.iloc[sub][FEATURES].reset_index(drop=True); y = t[TARGET].to_numpy()[sub]
        base = r2_score(y, pipe.predict(X))
        units = {**{f"group:{g}": cols for g, cols in PREDICTORS.items()}, **{c: [c] for c in FEATURES}}
        for u, cols in units.items():
            drops = []
            for r in range(5):
                Xp = X.copy(); perm = rng.permutation(len(X)); Xp[cols] = X[cols].to_numpy()[perm]
                drops.append(base - r2_score(y, pipe.predict(Xp)))
            rows.append({"design": design, "unit": u, "is_group": u.startswith("group:"), "r2_drop_mean": float(np.mean(drops)),
                         "r2_drop_sd": float(np.std(drops)), "base_R2": base})
        log(f"importance {design}: done (base R2 {base:.3f})")
    pd.DataFrame(rows).to_csv(f, index=False)

def stage_screen(t, force=False):
    f = OUT / "screen_random_split.csv"
    if f.exists() and not force: return pd.read_csv(f)
    tr, te, _ = split_random(t, 42)
    Xtr, Xte, ytr, yte = t.iloc[tr][FEATURES], t.iloc[te][FEATURES], t[TARGET].to_numpy()[tr], t[TARGET].to_numpy()[te]
    sc = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
    models = {"Linear Regression": sc(LinearRegression()), "Ridge": sc(Ridge()), "Lasso": sc(Lasso()),
              "Elastic Net": sc(ElasticNet()), "k-Nearest Neighbors": sc(KNeighborsRegressor()),
              "Gradient Boosting": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(random_state=42)),
              **{k: make_pipeline(SimpleImputer(strategy="median"), v) for k, v in tree_models(42).items()}}
    rows = []
    for name, pipe in models.items():
        t0 = time.time(); pipe.fit(Xtr, ytr)
        rows.append({"model": name, "fit_seconds": time.time() - t0, "train_rows": len(ytr), **metrics(yte, pipe.predict(Xte))})
        log(f"screen {name}: R2={rows[-1]['R2']:.3f}")
    sub = np.random.default_rng(42).choice(len(tr), size=min(20000, len(tr)), replace=False)
    svr = TransformedTargetRegressor(regressor=sc(SVR(kernel="rbf")), transformer=StandardScaler())
    t0 = time.time(); svr.fit(Xtr.iloc[sub], ytr[sub])
    rows.append({"model": "Support Vector Regression", "fit_seconds": time.time() - t0, "train_rows": len(sub), **metrics(yte, svr.predict(Xte))})
    log(f"screen SVR (20k subsample, scaled target): R2={rows[-1]['R2']:.3f}")
    s = pd.DataFrame(rows); s.to_csv(f, index=False); return s

def stage_sensitivity(t, force=False):
    f1, f2, f3 = OUT / "sensitivity_min_records.csv", OUT / "sensitivity_ratio_per_well.csv", OUT / "sensitivity_origin_global.csv"
    if f1.exists() and f2.exists() and f3.exists() and not force: return
    rows = []
    size = t.groupby(WELL)[WELL].transform("size")
    for mr in [3, 9, 27, 54, 81]:
        sub = t[size >= mr].reset_index(drop=True)          # stated design: wells must have >= mr well-months to enter
        tr, te, info = split_temporal_per_well(sub)
        for name, model in tree_models(42).items():
            _, p, _ = fit_predict(model, sub.iloc[tr][FEATURES], sub[TARGET].to_numpy()[tr], sub.iloc[te][FEATURES])
            rows.append({"min_records": mr, "model": name, "wells": sub[WELL].nunique(), "n_train": len(tr),
                         "horizon_median_y": info["horizon_median_y"], **metrics(sub[TARGET].to_numpy()[te], p)})
            log(f"sens min_records={mr} {name}: R2={rows[-1]['R2']:.3f}")
    pd.DataFrame(rows).to_csv(f1, index=False)
    for fn, grid, col, fout in [(split_temporal_per_well, [.1, .2, .3, .4, .5, .6, .7, .8, .9], "train_fraction", f2),
                                (split_temporal_global, [50, 60, 70, 80, 90], "origin_percentile", f3)]:
        rows = []
        for v in grid:
            tr, te, info = fn(t, frac_train=v) if col == "train_fraction" else fn(t, pct=v)
            for name, model in tree_models(42).items():
                _, p, _ = fit_predict(model, t.iloc[tr][FEATURES], t[TARGET].to_numpy()[tr], t.iloc[te][FEATURES])
                rows.append({col: v, "model": name, "n_train": len(tr), "pct_train_of_table": 100 * len(tr) / len(t),
                             "horizon_median_y": info["horizon_median_y"], **metrics(t[TARGET].to_numpy()[te], p)})
                log(f"sens {col}={v} {name}: R2={rows[-1]['R2']:.3f}")
        pd.DataFrame(rows).to_csv(fout, index=False)

def stage_sensitivity_baselines(t, force=False):
    """Admissible baselines on exactly the partitions of the three sensitivity analyses (deterministic splits)."""
    f = OUT / "sensitivity_baselines.csv"
    if f.exists() and not force: return pd.read_csv(f)
    rows = []
    size = t.groupby(WELL)[WELL].transform("size")
    for mr in [3, 9, 27, 54, 81]:
        sub = t[size >= mr].reset_index(drop=True)
        tr, te, _ = split_temporal_per_well(sub)
        for b, mm in baselines(sub, tr, te, "temporal_per_well").items():
            rows.append({"analysis": "min_records", "value": mr, "model": b, **mm})
    for v in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        tr, te, _ = split_temporal_per_well(t, frac_train=v)
        for b, mm in baselines(t, tr, te, "temporal_per_well").items():
            rows.append({"analysis": "train_fraction", "value": v, "model": b, **mm})
    for v in [50, 60, 70, 80, 90]:
        tr, te, _ = split_temporal_global(t, pct=v)
        for b, mm in baselines(t, tr, te, "temporal_global").items():
            rows.append({"analysis": "origin_percentile", "value": v, "model": b, **mm})
    out = pd.DataFrame(rows); out.to_csv(f, index=False); log(f"sensitivity baselines: {len(out)} rows"); return out

HORIZON_BINS = [0, 1, 2, 5, 10, 50]
def stage_horizon(t, force=False):
    """Error by horizon (years from the origin to the test month) for the learners (seed 42 predictions of stage_main)
    and the admissible baselines, on the identical chronological partitions."""
    f = OUT / "horizon_errors.csv"
    if f.exists() and not force: return pd.read_csv(f)
    rows = []
    for design in ["temporal_per_well", "temporal_global"]:
        tr, te, info = DESIGNS[design](t, None)
        p = pd.read_parquet(OUT / "predictions" / f"main_{design}_seed42.parquet")
        assert np.array_equal(p.row.to_numpy(), te), "saved predictions must match the recomputed partition"
        for b, v in baseline_predictions(t, tr, te, design).items(): p[b] = v
        if design == "temporal_global":
            p["origin"] = pd.Timestamp(info["origin"]) - pd.DateOffset(months=1)   # last training month, as per well
        else:
            last_tr = t.iloc[tr].groupby(WELL)["month"].max()
            p["origin"] = p.well.map(last_tr)
        p["horizon_y"] = (pd.to_datetime(p.month) - pd.to_datetime(p.origin)).dt.days / 365.25
        p["bin"] = pd.cut(p.horizon_y, HORIZON_BINS, right=False)
        for b, g in p.groupby("bin", observed=True):
            for mdl in [c for c in p.columns if c not in ("row", "well", "month", "y_true", "origin", "horizon_y", "bin")]:
                rows.append({"design": design, "horizon_bin": str(b), "h_lo": b.left, "h_hi": b.right, "model": mdl,
                             **metrics(g.y_true, g[mdl])})
    out = pd.DataFrame(rows); out.to_csv(f, index=False); log(f"horizon errors: {len(out)} rows"); return out

def stage_spatial_bands(t, force=False):
    """Spatial-block error of learners and baselines by latitude band of the test wells, every seed."""
    f = OUT / "spatial_bands.csv"
    if f.exists() and not force: return pd.read_csv(f)
    rows = []
    for seed in SEEDS:
        tr, te, _ = split_spatial(t, seed)
        p = pd.read_parquet(OUT / "predictions" / f"main_spatial_seed{seed}.parquet")
        assert np.array_equal(p.row.to_numpy(), te), "saved predictions must match the recomputed partition"
        for b, v in baseline_predictions(t, tr, te, "spatial").items(): p[b] = v
        p["band"] = pd.cut(t.iloc[te]["well_lat"].to_numpy(), LAT_BANDS)
        models = [c for c in p.columns if c not in ("row", "well", "month", "y_true", "band")]
        for band, g in p.groupby("band", observed=True):
            for mdl in models:
                rows.append({"seed": seed, "band": str(band), "lat_lo": band.left, "lat_hi": band.right, "model": mdl,
                             "n_wells": g.well.nunique(), **metrics(g.y_true, g[mdl])})
    out = pd.DataFrame(rows); out.to_csv(f, index=False); log(f"spatial bands: {len(out)} rows"); return out

def month_folds(months, n_splits=5):
    """Forward chaining on whole calendar months: each fold trains on earlier months, validates on the next block."""
    um = np.sort(np.unique(months)); blocks = np.array_split(um, n_splits + 1); folds = []
    for k in range(1, n_splits + 1):
        tr_m = np.concatenate(blocks[:k]); va_m = blocks[k]
        folds.append((np.where(np.isin(months, tr_m))[0], np.where(np.isin(months, va_m))[0]))
    return folds, [(str(pd.Timestamp(b[0]).date()), str(pd.Timestamp(b[-1]).date())) for b in blocks]

def stage_tuning(t, force=False):
    """Tune on the TRAINING partition only; compare shuffled 5-fold with forward chaining; both temporal designs."""
    f = OUT / "tuning.csv"
    if f.exists() and not force: return pd.read_csv(f)
    grids = {"Decision Tree": {"m__max_depth": [None, 20, 10], "m__min_samples_split": [2, 10], "m__min_samples_leaf": [1, 4, 16]},
             "Random Forest": {"m__max_depth": [None, 20], "m__max_features": [1.0, 0.33], "m__min_samples_leaf": [1, 5]},
             "Extra Trees": {"m__max_depth": [None, 20], "m__max_features": [1.0, 0.33], "m__min_samples_leaf": [1, 5]}}
    rows, fold_info = [], {}
    for design in ["temporal_global", "temporal_per_well"]:
        tr, te, _ = DESIGNS[design](t, None)
        Xtr, ytr = t.iloc[tr][FEATURES], t[TARGET].to_numpy()[tr]
        Xte, yte = t.iloc[te][FEATURES], t[TARGET].to_numpy()[te]
        folds, blocks = month_folds(t.iloc[tr]["month"].to_numpy()); fold_info[design] = blocks
        for name, model in tree_models(42).items():
            _, p, _ = fit_predict(model, Xtr, ytr, Xte)                      # same order and seed as stage_main
            rows.append({"design": design, "model": name, "scheme": "Default", "best_params": "", "cv_RMSE": np.nan,
                         "search_seconds": 0.0, **metrics(yte, p)})
            base = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", clone(model))])
            for sname, cv in {"5-fold (shuffled)": KFold(5, shuffle=True, random_state=42), "Forward chaining (5 folds)": folds}.items():
                t0 = time.time()
                gs = GridSearchCV(base, grids[name], cv=cv, scoring="neg_root_mean_squared_error",
                                  n_jobs=1 if name != "Decision Tree" else N_JOBS)
                gs.fit(Xtr, ytr)
                rows.append({"design": design, "model": name, "scheme": sname, "best_params": json.dumps(gs.best_params_),
                             "cv_RMSE": -gs.best_score_, "search_seconds": time.time() - t0, **metrics(yte, gs.predict(Xte))})
                log(f"tuning {design} {name} {sname}: {gs.best_params_} test R2={rows[-1]['R2']:.3f} ({time.time()-t0:.0f}s)")
                pd.DataFrame(rows).to_csv(f.with_suffix(".partial.csv"), index=False)
    json.dump({"grids": {k: {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in grids.items()},
               "forward_chaining_blocks": fold_info}, open(OUT / "tuning_design.json", "w"), indent=1)
    out = pd.DataFrame(rows); out.to_csv(f, index=False); return out

def write_manifest():
    import matplotlib, scipy
    man = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
           "features": FEATURES, "n_features": len(FEATURES), "predictor_groups": PREDICTORS,
           "design": DESIGN, "seeds": SEEDS, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(man, open(OUT / "run_manifest.json", "w"), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "main", "nocoord", "nomonthly", "lononly", "importance", "screen", "sensitivity", "sensbase",
                                                    "horizon", "spatialbands", "tuning"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); write_manifest()
    t = stage_prepare(force=a.force and a.stage == "prepare")
    if a.stage in ("all", "main"): stage_main(t, force=a.force)
    if a.stage in ("all", "importance"): stage_importance(t, force=a.force)
    if a.stage in ("all", "screen"): stage_screen(t, force=a.force)
    if a.stage in ("all", "sensitivity"): stage_sensitivity(t, force=a.force)
    if a.stage in ("all", "sensbase"): stage_sensitivity_baselines(t, force=a.force)
    if a.stage in ("all", "horizon"): stage_horizon(t, force=a.force)
    for tag, feats in ABLATIONS.items():
        if a.stage in ("all", tag): stage_main(t, force=a.force, features=feats, tag=tag)
    if a.stage in ("all", "spatialbands"): stage_spatial_bands(t, force=a.force)
    if a.stage in ("all", "tuning"): stage_tuning(t, force=a.force)
    log(f"done stage={a.stage}")

if __name__ == "__main__":
    main()
