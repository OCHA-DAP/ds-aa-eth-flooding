# Ethiopia Flood Anticipatory Action

Trigger-design work for riverine flooding in Ethiopia's Somali region
(Shabelle + Genale/Dawa basins), following the approach used for Somalia
(`ds-aa-som-floods`): GloFAS return-period thresholds validated against
FloodScan-derived flood events, plus a live forecast dashboard.

## Data

All data lives on Azure blob (`projects/dev`, prefix `ds-aa-eth-flooding/`).
Notebooks and scripts read/write blob via `ocha_stratus`; nothing large is
committed here.

Monitoring points (8): the 3 official GloFAS reporting points (Shabelle at
Gode G1904, Genale/Juba at Dolow, upper Genale) and 5 ad-hoc grid points
(4 on the Shabelle, 1 on the Webe Gestro in Afder). All are channel-snapped to
GloFAS's 0.05 degree river network: the raw reporting-point coordinates sit
on dry cells (see `analysis/resnap_reporting_points.py`).

| blob path | content |
|---|---|
| `processed/glofas/glofas_discharge.parquet` | v4 reanalysis + reforecast, 5 grid points (ensemble reforecast incomplete: control only usable) |
| `processed/glofas/glofas_discharge_reporting_points.parquet` | v4 reanalysis (2003 to present) + full ensemble reforecast (2003-2023, Gu + Deyr months, leads 1-7 d), 3 reporting points, channel-snapped |
| `processed/glofas/glofas_discharge_v5.parquet` | v5 reanalysis 1980-2026, all 8 points |
| `processed/glofas/glofas_return_periods.parquet` | Gumbel + empirical thresholds per version, station, season (annual, Gu, Kiremt, Deyr), RP 2-10 |
| `processed/glofas/glofas_forecast_latest.parquet` | latest operational forecast (51-member, leads 1-10 d), overwritten each fetch |
| `processed/floodscan/floodscan_somali_eth.csv` | FloodScan flooded extent per admin2 zone, 1998 to present |
| `processed/floodscan/floodscan_event_catalogue.parquet` | flood events per zone at FloodScan RP2/RP3 levels, tagged by season |
| `processed/comparison/*.parquet` | validation scores (reanalysis + reforecast vs FloodScan events) |

## Analysis

Run order:

1. `analysis/01_glofas_return_periods.ipynb` : threshold fitting (v4 + v5, seasonal)
2. `analysis/02_floodscan_events_validation.ipynb` : FloodScan event catalogue + POD/FAR of GloFAS RP crossings
3. `analysis/03_reforecast_trigger_skill.ipynb` : ensemble trigger skill by lead time (v4 reforecast)

Key findings so far:

- The Jul-Sep highland rains dominate the annual discharge maxima, so Gu and
  Deyr triggers need seasonal thresholds: annual ones are blind to both.
- GloFAS v4 misses the 2023 Deyr floods entirely at Gode (below RP2), consistent
  with the Somalia finding that v4 is poorly calibrated on the Shabelle. v5
  reanalysis fixes this (Gode Deyr POD 0.00 -> 0.67 vs FloodScan events), BUT
  the operational forecast and the GloFAS map viewer thresholds are still
  v4-scale (verified 2026-08-31: viewer RP1.5 ~ 200 m3/s at the Webe Gestro
  point = v4 annual RP2 236, vs v5's 59), so the live dashboard pins v4
  thresholds and carries a tripwire for the v5 operational upgrade
  (`THRESHOLD_VERSION_PIN` in `analysis/make_dashboard.py`).
- Reforecast skill is flat across leads 1-7 days (initial-condition memory), so
  the trigger lead time can be chosen operationally rather than by skill decay.

## Monitoring dashboard

`analysis/make_dashboard.py` builds the Deyr 2026 flood watch from the latest
operational forecast (`analysis/fetch_glofas_forecast_live.py`): station map,
per-station ensemble fan charts vs seasonal RP2/RP3/RP5 levels, and the
monitored Deyr conditions. Output: `docs/index.html`, served via GitHub Pages
at https://ocha-dap.github.io/ds-aa-eth-flooding/.

Daily refresh runs on GitHub: `.github/workflows/deyr-monitoring.yml`
(11:15 UTC) fetches the latest forecast from EWDS, writes the extracted data
to blob, rebuilds the page and commits it. Repository secrets required:
`EWDS_API_KEY`, `DSCI_AZ_BLOB_DEV_SAS`, `DSCI_AZ_BLOB_DEV_SAS_WRITE`. The
page carries an update log (one line per run) so missed days are visible.

## Fetch scripts

- `analysis/fetch_glofas.py` / `fetch_glofas_reporting_points.py` : original v4
  EWDS fetches (box netcdfs -> blob -> extracted parquets)
- `analysis/resnap_reporting_points.py` : channel-snap fix, re-extracts the
  reporting points from the raw netcdfs already on blob
- `analysis/fetch_glofas_v5_reanalysis.py` : v5 reanalysis 1980-2026 for the
  8-station box
- `analysis/fetch_glofas_forecast_live.py` : latest operational forecast for the
  dashboard (re-run to refresh)

Environment: needs `ocha_stratus`, `cdsapi` (EWDS key in `~/.cdsapirc`),
`xarray`, `geopandas`, `scipy`. Blob SAS env vars must be set.
