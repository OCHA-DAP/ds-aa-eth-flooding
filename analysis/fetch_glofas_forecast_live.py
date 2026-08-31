"""
Fetch the latest operational GloFAS forecast (cems-glofas-forecast,
system_version=operational, 51-member ECMWF-ENS driven) for the Somali-region
box, extract at the station channel cells, and upload for the dashboard.

The latest available issue date is discovered from the collection's
constraints.json (operational forecasts land on EWDS with ~1 day latency).
Extraction cells: stations_v5_cells.csv if present (the operational system
runs GloFAS v5, so v5-snapped cells are the right ones), else the v4
channel cells.

Outputs on blob (projects/dev):
- raw/glofas/forecast_live/fc_<issue>_<product>.nc
- processed/glofas/glofas_forecast_latest.parquet (overwritten each run:
  the dashboard always shows the newest issue; raw netcdfs accumulate)

Re-runnable: skips products whose raw blob for the latest issue already
exists, but still rebuilds the parquet from local/blob raw files.
"""

import logging
import os
from pathlib import Path

import cdsapi
import ocha_stratus as stratus
import pandas as pd
import requests
import xarray as xr
import yaml

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/glofas/forecast_live"
OUT_PARQUET = f"{PROJECT_PREFIX}/processed/glofas/glofas_forecast_latest.parquet"

SCRATCH_DIR = Path(
    os.environ.get("GLOFAS_SCRATCH", str(Path(__file__).resolve().parent / "scratch_glofas_rp"))
) / "forecast_live"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_forecast_live")
logging.getLogger("azure").setLevel(logging.WARNING)

AREA = [6.575, 39.95, 3.65, 45.325]  # N, W, S, E: all 8 stations + 0.5deg
LEADTIME_HOURS = [str(h) for h in range(24, 24 * 10 + 1, 24)]  # 1-10 days
PRODUCTS = ["control_forecast", "ensemble_perturbed_forecasts"]
CONSTRAINTS_URL = (
    "https://ewds.climate.copernicus.eu/api/catalogue/v1/collections/"
    "cems-glofas-forecast/constraints.json"
)


def latest_issue_date() -> pd.Timestamp:
    cons = requests.get(CONSTRAINTS_URL, timeout=60).json()
    dates = []
    for block in cons:
        if "operational" not in block.get("system_version", []):
            continue
        for y in block.get("year", []):
            for m in block.get("month", []):
                for d in block.get("day", []):
                    try:
                        dates.append(pd.Timestamp(int(y), int(m), int(d)))
                    except ValueError:
                        pass
    return max(dates)


def load_cells() -> pd.DataFrame:
    try:
        cells = stratus.load_csv_from_blob(
            f"{PROJECT_PREFIX}/raw/glofas/stations_v5_cells.csv", stage=STAGE
        ).rename(columns={"v5_lon": "cell_lon", "v5_lat": "cell_lat"})
        log.info("using v5 channel cells")
    except Exception:
        grid = stratus.load_csv_from_blob(
            f"{PROJECT_PREFIX}/raw/glofas/stations.csv", stage=STAGE
        ).rename(columns={"lon": "cell_lon", "lat": "cell_lat"})
        snapped = stratus.load_csv_from_blob(
            f"{PROJECT_PREFIX}/raw/glofas/stations_reporting_points_somali_snapped.csv",
            stage=STAGE,
        ).rename(columns={"snapped_lon": "cell_lon", "snapped_lat": "cell_lat"})
        cells = pd.concat(
            [grid[["station_id", "cell_lon", "cell_lat"]],
             snapped[["station_id", "cell_lon", "cell_lat"]]],
            ignore_index=True,
        )
        log.info("v5 cells not on blob yet: using v4 channel cells")
    return cells[["station_id", "cell_lon", "cell_lat"]]


def discharge_var(ds: xr.Dataset) -> str:
    return next(v for v in ds.data_vars if "dis" in v.lower())


def extract(nc_path: Path, cells: pd.DataFrame, product: str) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    var = discharge_var(ds)
    frames = []
    for _, st in cells.iterrows():
        sub = ds[var].sel(latitude=st["cell_lat"], longitude=st["cell_lon"], method="nearest")
        df = sub.to_dataframe(name="discharge").reset_index()
        df["station_id"] = st["station_id"]
        df["product_type"] = product
        frames.append(df)
    ds.close()
    out = pd.concat(frames, ignore_index=True)
    if "forecast_period" in out.columns:
        out["leadtime_days"] = out["forecast_period"].dt.days
        out = out.rename(columns={"forecast_reference_time": "issued_time"})
    keep = [c for c in [
        "station_id", "product_type", "issued_time", "valid_time",
        "leadtime_days", "number", "latitude", "longitude", "discharge",
    ] if c in out.columns]
    return out[keep]


def main() -> None:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key=cfg["key"])

    issue = latest_issue_date()
    log.info(f"latest operational issue on EWDS: {issue:%Y-%m-%d}")
    cells = load_cells()

    existing = set(
        stratus.list_container_blobs(name_starts_with=f"{RAW_PREFIX}/", stage=STAGE, container_name=CONTAINER)
    )
    frames = []
    for product in PRODUCTS:
        stem = f"fc_{issue:%Y%m%d}_{product}"
        local = SCRATCH_DIR / f"{stem}.nc"
        blob_name = f"{RAW_PREFIX}/{stem}.nc"
        if not local.exists():
            if blob_name in existing:
                local.write_bytes(
                    stratus.load_blob_data(blob_name, stage=STAGE, container_name=CONTAINER)
                )
                log.info(f"{stem}: pulled raw from blob")
            else:
                client.retrieve("cems-glofas-forecast", {
                    "system_version": "operational",
                    "hydrological_model": "lisflood",
                    "product_type": product,
                    "variable": "river_discharge_in_the_last_24_hours",
                    "year": str(issue.year),
                    "month": f"{issue.month:02d}",
                    "day": f"{issue.day:02d}",
                    "leadtime_hour": LEADTIME_HOURS,
                    "data_format": "netcdf",
                    "area": AREA,
                }, str(local))
                with open(local, "rb") as f:
                    stratus.upload_blob_data(
                        f.read(), blob_name, stage=STAGE, container_name=CONTAINER,
                        content_type="application/x-netcdf",
                    )
                log.info(f"{stem}: downloaded + uploaded")
        frames.append(extract(local, cells, product))

    combined = pd.concat(frames, ignore_index=True)
    stratus.upload_parquet_to_blob(combined, OUT_PARQUET, stage=STAGE)
    log.info(f"{len(combined)} rows ({issue:%Y-%m-%d} issue) -> {OUT_PARQUET}")


if __name__ == "__main__":
    main()
