# Chile_groundwater_ML

Code for *Matching Validation Design to the Prediction Task in Groundwater Machine
Learning: An Open National Benchmark for Chile* (manuscript submitted to *Data*, MDPI).

The study builds an analysis-ready table of monthly depth-to-water (DTW) values for the
Chilean national monitoring network, scores three tree-based learners under five
established train/test splitting designs, and compares every learner with simple
baselines computed on the same partitions. Five scripts in `scripts/` produce every number,
table and figure in the paper from the source file.

**Status:** under submission; the article DOI will be added here.

- Data (analysis table, split assignments, predictions, results): https://doi.org/10.5281/zenodo.23003804
- Code archive (this repository, v1.0.2): https://doi.org/10.5281/zenodo.23003808

---

## What the analysis does

| Design | Train–test dependence removed | Baselines reported |
|---|---|---|
| Random (70/30 of well-months) | none | training mean, well training mean, temporal interpolation (same well), neighbor interpolation |
| Well-based (co-location groups held out) | within-well | training mean, neighbor interpolation |
| Spatial block (10 x 10 grid, whole cells held out, no buffer) | within-well and short-range between-well | training mean, neighbor interpolation |
| Chronological, per well (first 70% of each well's record trains) | later records of each well | training mean, well training mean, last observation |
| Chronological, global origin (one calendar origin) | all information after the origin | training mean, well training mean, last observation |

Learners: Decision Tree, Random Forest (100 trees), Extra Trees (100 trees), scikit-learn
defaults otherwise; five seeds per design. Skill is reported as `1 - MSE/MSE_ref` against the
best admissible baseline of each design.

Neighbor interpolation is the inverse-distance mean of the training means of the five
geographically nearest other wells; temporal interpolation is linear in time between the same
well's training well-months.

Additional stages: grouped permutation importance, drop-column ablations (without coordinates,
without monthly climate, longitude only), a ten-learner screen, sensitivity to record length,
training fraction and split position (with baselines on the same partitions), error by horizon
and by latitude band, and hyperparameter tuning with shuffled versus forward-chaining folds.

## Repository layout

```
scripts/
  rift_pipeline.py              the whole analysis: data preparation, splits, models, baselines, all stages
  fetch_terraclimate_bbox.py    re-extracts TerraClimate for each record's own calendar month (public THREDDS server)
  make_figures.py               draws Figures 2-8 and S1-S7 from scripts/outputs/
  make_figure1.py               draws the workflow diagram (Figure 1) from the pipeline's provenance files
  make_facts.py                 derives every number and table reported in the paper (outputs/facts.json, outputs/tables/)
  deposit.json                  the Zenodo DOIs of the data and code records, used in the paper's Data Availability Statement
extraction/
  01_data_integration_google_earth_engine.ipynb   how the static predictors in the source file were extracted
legacy/                          notebooks from an earlier, superseded analysis (see legacy/README.md)
data/
  README.md                      where to obtain the inputs
  sample/                        a 2,000-row sample of the source file for smoke tests
requirements.txt
```

## Reproducing the results

1. Obtain the inputs (see [`data/README.md`](data/README.md)) and place them in `data/raw/`,
   or point to them with the environment variables `RIFT_SOURCE_CSV` (source file) and
   `RIFT_SHAC_SHP` (DGA SHAC polygons, used only as a map background).
2. Create the environment: `pip install -r requirements.txt` (Python 3.13).
3. Run, from `scripts/`:

```bash
python fetch_terraclimate_bbox.py        # about 410 small NetCDF requests; files are cached
python rift_pipeline.py --stage all       # several hours on a laptop; stages cache their outputs
python make_figures.py
python make_figure1.py
python make_facts.py
```

`fetch_terraclimate_bbox.py` downloads the current TerraClimate release, whose values can change
between releases. To reproduce the published numbers exactly, copy
`terraclimate_by_location_month.parquet` from the Zenodo data record into `scripts/outputs/` and
skip that step.

Outputs are written to `scripts/outputs/` (metrics, split summaries, per-row split assignments in
`split_assignments_main.parquet`, predictions for every test well-month, provenance, a run manifest with package
versions and design settings, `facts.json` and `tables/`) and `scripts/figures/`. A single stage can be rerun
with, for example, `python rift_pipeline.py --stage tuning --force`.

The source file's SHA-256 checksum is recorded in `outputs/data_provenance.json`; the value
used for the paper begins `369a41523a64`.

## What the analysis does not include

- No geostatistical model (kriging with external drift, regression kriging) and no sequence
  or autoregressive model; the baselines are the simple ones listed above.
- No buffered or distance-matched (k-fold nearest-neighbor distance matching) cross-validation and no area-of-applicability
  analysis; the spatial block design uses one fixed, unbuffered grid.
- Hyperparameter tuning only for the two chronological designs.
- No lithology, aquifer properties, abstraction or snow variables, because the public
  archives do not contain them.

## Citation

See [`CITATION.cff`](CITATION.cff). The groundwater observations come from
Venegas-Quiñones et al. (2024), *Scientific Data* 11, 170,
https://doi.org/10.1038/s41597-023-02895-5, with the dataset at
https://doi.org/10.17605/OSF.IO/DS3A8.

## License

MIT (see `LICENSE`).
