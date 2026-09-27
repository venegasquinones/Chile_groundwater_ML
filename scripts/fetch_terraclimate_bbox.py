"""
Re-extract TerraClimate (Abatzoglou et al. 2018) monthly values for every well location, keyed to the record's OWN
calendar month. Source: public TerraClimate THREDDS server (Northwest Knowledge Network, University of Idaho),
NetCDF Subset Service, one request per variable per year over the wells' bounding box, then nearest-cell sampling
at each well location. Values are converted to physical units with each file's scale factor and offset.

    python fetch_terraclimate_bbox.py            # fetch + sample + validate, resumable (files cached)
"""
import json, time, urllib.request
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import netcdf_file

HERE = Path(__file__).resolve().parent
TMP = HERE / "terraclimate_tmp"; TMP.mkdir(exist_ok=True)
(HERE / "outputs").mkdir(exist_ok=True)
OUTF = HERE / "outputs" / "terraclimate_by_location_month.parquet"
CSV = Path(os.environ.get("RIFT_SOURCE_CSV", HERE.parent / "data" / "raw" /
                          "groundwater_chile_and_elevation_dataset_2025_with_GEE_7_2_25.csv"))  # source file from the Zenodo deposit
BASE = "https://thredds.northwestknowledge.net/thredds/ncss/grid"
VARS = {"pr": ("ppt", "ppt"), "tmmn": ("tmin", "tmin"), "tmmx": ("tmax", "tmax"), "aet": ("aet", "aet"),
        "pet": ("pet", "pet"), "def": ("def", "def"), "pdsi": ("PDSI", "PDSI"), "srad": ("srad", "srad"),
        "vpd": ("vpd", "vpd"), "vs": ("ws", "ws")}
YEARS = range(1981, 2022)
BOX = dict(north=-17.5, south=-42.0, west=-74.0, east=-68.0)
UA = {"User-Agent": "RIFT-reanalysis (hector.venegasquinones@mines.edu)"}
SCALES = {}

def locations():
    raw = pd.read_csv(CSV, encoding="latin-1", low_memory=False, usecols=["Longitude_GCS_WGS_1984", "Latitude_GCS_WGS_1984"])
    L = raw.round(5).drop_duplicates().rename(columns={"Longitude_GCS_WGS_1984": "lon", "Latitude_GCS_WGS_1984": "lat"})
    L = L[(L.lon.between(BOX["west"], BOX["east"])) & (L.lat.between(BOX["south"], BOX["north"]))]
    return L.reset_index(drop=True)

def fetch_year(band, year):
    ds, var = VARS[band]
    f = TMP / f"{band}_{year}.nc"
    if f.exists() and f.stat().st_size > 1000:
        return f
    url = (f"{BASE}/agg_terraclimate_{ds}_1950_CurrentYear_GLOBE.nc?var={var}&north={BOX['north']}&south={BOX['south']}"
           f"&west={BOX['west']}&east={BOX['east']}&horizStride=1&time_start={year}-01-01T00:00:00Z"
           f"&time_end={year}-12-31T00:00:00Z&accept=netcdf")
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                b = r.read()
            if b[:3] != b"CDF":
                raise RuntimeError("not a NetCDF-3 response")
            f.write_bytes(b)
            return f
        except Exception as e:
            err = e; time.sleep(5 * (k + 1))
    raise RuntimeError(f"{band} {year}: {err}")

def sample(f, band, L):
    var = VARS[band][1]
    nc = netcdf_file(str(f), "r", mmap=False)
    v = nc.variables[var]
    data = np.array(v[:]).astype("float64")                       # (time, lat, lon), packed
    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        data[data == float(fill)] = np.nan
    SCALES[band] = (float(getattr(v, "scale_factor", 1.0)), float(getattr(v, "add_offset", 0.0)))
    data = data * SCALES[band][0] + SCALES[band][1]                # physical units
    lat = np.array(nc.variables["lat"][:]); lon = np.array(nc.variables["lon"][:])
    days = np.array(nc.variables["time"][:])
    nc.close()
    ilat = np.abs(lat[None, :] - L.lat.values[:, None]).argmin(1)
    ilon = np.abs(lon[None, :] - L.lon.values[:, None]).argmin(1)
    vals = data[:, ilat, ilon]                                       # (time, n_points)
    months = (pd.Timestamp("1900-01-01") + pd.to_timedelta(days, unit="D")).to_period("M").to_timestamp()
    return pd.DataFrame({"lon": np.tile(L.lon.values, len(months)), "lat": np.tile(L.lat.values, len(months)),
                         "month": np.repeat(months, len(L)), band: vals.ravel()})

def main():
    L = locations(); print("locations:", len(L), flush=True); t0 = time.time()
    frames = {b: [] for b in VARS}; n = 0
    for year in YEARS:
        for band in VARS:
            f = fetch_year(band, year); frames[band].append(sample(f, band, L)); n += 1
        print(f"{year} done ({n} files, {time.time()-t0:.0f}s)", flush=True)
    tab = None
    for band in VARS:
        d = pd.concat(frames[band], ignore_index=True)
        tab = d if tab is None else tab.merge(d, on=["lon", "lat", "month"], how="outer")
    tab.to_parquet(OUTF)
    print("wrote", OUTF, tab.shape, flush=True)
    # ---- validation: at the month GEE reported, do the two extractions agree?
    tc = [f"terraclim_{b}_value" for b in VARS] + [f"terraclim_{b}_value_scale" for b in VARS]
    raw = pd.read_csv(CSV, encoding="latin-1", low_memory=False,
                      usecols=["Longitude_GCS_WGS_1984", "Latitude_GCS_WGS_1984", "terraclim_rs_date"] + tc)
    raw["lon"] = raw.Longitude_GCS_WGS_1984.round(5); raw["lat"] = raw.Latitude_GCS_WGS_1984.round(5)
    raw["month"] = pd.to_datetime(raw.terraclim_rs_date, errors="coerce").dt.to_period("M").dt.to_timestamp()
    m = raw.merge(tab, on=["lon", "lat", "month"], how="inner")
    GEE_SCALE = {"pr": 1, "tmmn": 0.1, "tmmx": 0.1, "aet": 0.1, "pet": 0.1, "def": 0.1, "pdsi": 0.01, "srad": 0.1,
                 "vpd": 0.01, "vs": 0.01}   # GEE catalog IDAHO_EPSCOR/TERRACLIMATE band scales
    gee = {b: m[f"terraclim_{b}_value"] * GEE_SCALE[b] for b in VARS}
    mad = {b: float((m[b] - gee[b]).abs().median()) for b in VARS}
    corr = {b: float(np.corrcoef(gee[b], m[b])[0, 1]) for b in VARS}
    means = {b: [float(gee[b].mean()), float(m[b].mean())] for b in VARS}
    csv_scale = {b: float(m[f"terraclim_{b}_value_scale"].iloc[0]) for b in VARS}
    rep = {"rows_compared": len(m), "note": "physical units; GEE asset (older TerraClimate release) vs THREDDS current release "
           "at the GEE-reported month", "median_abs_diff_physical": mad, "pearson_r": corr,
           "mean_gee_vs_thredds": means, "csv_value_scale_column": csv_scale, "gee_catalog_scale": GEE_SCALE,
           "file_packing_scale_offset": SCALES, "seconds": time.time() - t0, "source": BASE, "bbox": BOX,
           "years": [min(YEARS), max(YEARS)], "locations": len(L),
           "accessed": min(time.strftime("%Y-%m-%d", time.localtime(f.stat().st_mtime)) for f in TMP.glob("*.nc"))}   # download date of the cached files
    json.dump(rep, open(HERE / "outputs" / "terraclimate_refetch_log.json", "w"), indent=1)
    print("rows compared:", len(m))
    for b in VARS:
        print(f"  {b:<5} median|diff|={mad[b]:.3f}  r={corr[b]:.4f}  mean GEE={means[b][0]:.3f} THREDDS={means[b][1]:.3f}  "
              f"CSV scale col={csv_scale[b]} GEE catalog={GEE_SCALE[b]}")

if __name__ == "__main__":
    main()
