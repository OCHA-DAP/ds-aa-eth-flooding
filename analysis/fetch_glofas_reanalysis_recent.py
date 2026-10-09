"""
Fetch the recent GloFAS v4 reanalysis (intermediate product) at the 14 watch
stations, so the page can show how each river has been running before the
10-day forecast.

EWDS publishes the v4 intermediate reanalysis a few days behind real time.
Each run fetches only the days not stored yet (up to the latest published
day), one small request per box and month, extracts the same channel cells
the forecast uses (raw/glofas/stations_v5_cells.csv), and keeps the last
DAYS_KEPT days in processed/glofas/glofas_reanalysis_recent.parquet
(station_id, valid_time, discharge).

Never fails the workflow: on any error it logs and leaves the stored file as
it is, and the page builds without the newest days.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import cdsapi
import ocha_stratus as stratus
import pandas as pd
import requests
import xarray as xr
import yaml

STAGE = "dev"
PROJECT_PREFIX = "ds-aa-eth-flooding"
OUT_BLOB = f"{PROJECT_PREFIX}/processed/glofas/glofas_reanalysis_recent.parquet"
DAYS_KEPT = 60
CONSTRAINTS_URL = (
    "https://ewds.climate.copernicus.eu/api/catalogue/v1/collections/"
    "cems-glofas-historical/constraints.json"
)
# N, W, S, E: the same boxes as fetch_glofas_forecast_live.py
BOXES = {
    "somali": [6.575, 39.95, 3.65, 45.325],
    "gambella": [8.7, 32.9, 6.2, 35.6],
    "omo_rift": [7.4, 35.6, 4.3, 38.4],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_reanalysis_recent")
logging.getLogger("azure").setLevel(logging.WARNING)


def box_of(lon: float) -> str:
    if lon >= 39.9:
        return "somali"
    return "gambella" if lon < 35.65 else "omo_rift"


def latest_available() -> pd.Timestamp:
    cons = requests.get(CONSTRAINTS_URL, timeout=60).json()
    dates = []
    for b in cons:
        if "version_4_0" not in b.get("system_version", []) or "intermediate" not in b.get("product_type", []):
            continue
        for y in b.get("year", []):
            for m in b.get("month", []):
                for d in b.get("day", []):
                    try:
                        dates.append(pd.Timestamp(int(y), int(m), int(d)))
                    except ValueError:
                        pass
    return max(dates)


def client() -> cdsapi.Client:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    return cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key=cfg["key"], quiet=True)


def main() -> None:
    try:
        stored = stratus.load_parquet_from_blob(OUT_BLOB, stage=STAGE)
        stored["valid_time"] = pd.to_datetime(stored["valid_time"])
    except Exception:
        stored = pd.DataFrame(columns=["station_id", "valid_time", "discharge"])
    latest = latest_available()
    start = latest - pd.Timedelta(days=DAYS_KEPT - 1)
    if len(stored):
        start = max(start, stored["valid_time"].max() + pd.Timedelta(days=1))
    if start > latest:
        log.info(f"reanalysis up to date (latest published {latest.date()})")
        return
    days = pd.date_range(start, latest, freq="D")
    log.info(f"fetching v4 intermediate reanalysis {days[0].date()} to {days[-1].date()}")

    cells = stratus.load_csv_from_blob(f"{PROJECT_PREFIX}/raw/glofas/stations_v5_cells.csv", stage=STAGE)
    cells["box"] = cells["v5_lon"].map(box_of)
    c = client()
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for box, area in BOXES.items():
        sts = cells[cells["box"] == box]
        for (y, m), part in pd.Series(days, index=days).groupby([days.year, days.month]):
            req = {
                "system_version": "version_4_0",
                "hydrological_model": "lisflood",
                "product_type": "intermediate",
                "variable": "average_river_discharge_in_the_last_24_hours",
                "timespan": "time_mean",
                "year": [str(y)],
                "month": [f"{m:02d}"],
                "day": [f"{d:02d}" for d in part.index.day],
                "data_format": "netcdf",
                "area": area,
            }
            out = tmp / f"{box}_{y}{m:02d}.nc"
            c.retrieve("cems-glofas-historical", req, str(out))
            ds = xr.open_dataset(out)
            var = next(v for v in ds.data_vars if "dis" in v.lower())
            tname = "valid_time" if "valid_time" in ds[var].dims else "time"
            for st in sts.itertuples():
                s = ds[var].sel(latitude=st.v5_lat, longitude=st.v5_lon, method="nearest").to_series()
                rows.append(pd.DataFrame({"station_id": st.station_id,
                                          "valid_time": pd.to_datetime(s.index.get_level_values(tname)),
                                          "discharge": s.values}))
            ds.close()
            log.info(f"  {box} {y}-{m:02d}: {len(part)} days")
    new = pd.concat(rows, ignore_index=True).dropna(subset=["discharge"])
    allr = pd.concat([stored, new], ignore_index=True)
    allr["valid_time"] = pd.to_datetime(allr["valid_time"]).dt.normalize()
    allr = allr.drop_duplicates(["station_id", "valid_time"], keep="last")
    allr = allr[allr["valid_time"] > latest - pd.Timedelta(days=DAYS_KEPT)]
    stratus.upload_parquet_to_blob(allr.sort_values(["station_id", "valid_time"]), OUT_BLOB, stage=STAGE)
    log.info(f"saved {OUT_BLOB}: {allr['station_id'].nunique()} stations, "
             f"{allr['valid_time'].min().date()} to {allr['valid_time'].max().date()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # the page must still build
        log.error(f"recent reanalysis not updated: {type(exc).__name__}: {str(exc)[:300]}")
        sys.exit(0)
