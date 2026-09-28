# Data

The input files are not stored in this repository. They are distributed through Zenodo
with a persistent identifier: https://doi.org/10.5281/zenodo.23003804 (`source_file.zip` holds the source file).

## Inputs

| File | Used by | Notes |
|---|---|---|
| `groundwater_chile_and_elevation_dataset_2025_with_GEE_7_2_25.csv` | `rift_pipeline.py`, `fetch_terraclimate_bbox.py` | Source file: the national DTW compilation with static predictors added through Google Earth Engine; 113,167 rows, 232 columns, latin-1 encoded, 284 MB. SHA-256 begins `369a41523a64`. |
| `INV_ACUIFEROS_SHAC.shp` (with its `.dbf`, `.shx`, `.prj`) | `make_figures.py` | Hydrogeological sectors (SHAC) of the Chilean water directorate (DGA), February 2023 version, from the DGA Mapoteca Digital (https://dga.mop.gob.cl/mapoteca/). Map background only. |

Place them in `data/raw/`, or set `RIFT_SOURCE_CSV` and `RIFT_SHAC_SHP` to their paths.

The monthly TerraClimate values are not an input: `fetch_terraclimate_bbox.py` downloads
them from the public THREDDS server of the Northwest Knowledge Network. The re-extracted
table used for the paper is part of the Zenodo deposit.

## Underlying sources

- Groundwater levels: Venegas-Quiñones et al. (2024), *Scientific Data* 11, 170,
  https://doi.org/10.1038/s41597-023-02895-5; dataset https://doi.org/10.17605/OSF.IO/DS3A8.
- Static predictors: Google Earth Engine catalog products (Copernicus DEM GLO-30, NASADEM,
  JAXA AW3D30 v3.2, WorldClim V1 BIO, MODIS MCD12Q1 v061).
  `extraction/01_data_integration_google_earth_engine.ipynb` documents the extraction of the
  Copernicus DEM, WorldClim, TerraClimate and MODIS values. The NASADEM (elevation, slope,
  aspect) and AW3D30 columns were already in the compilation, added earlier through Google
  Earth Engine; that extraction script is not included.
- Monthly climate: TerraClimate (Abatzoglou et al., 2018), https://doi.org/10.1038/sdata.2017.191.

## Sample

`sample/sample_2000_rows.csv.gz` contains the header and the first 2,000 rows of the source
file, byte for byte (latin-1 encoded, like the source file), for smoke tests. It is not
sufficient to reproduce any published result. `sample/columns.txt` lists the 232 columns and
their dtypes as read from the full source file (`pd.read_csv(..., encoding="latin-1", low_memory=False)`).
