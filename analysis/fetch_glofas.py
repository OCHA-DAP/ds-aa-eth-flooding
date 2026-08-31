"""
Fetch GloFAS reanalysis (cems-glofas-historical) and reforecast
(cems-glofas-reforecast) river discharge from the CEMS Early Warning Data
Store (EWDS) at the Shabelle river stations, and push raw + extracted
outputs to Azure Blob Storage via ocha_stratus.

GloFAS reforecast is only available 1999 -> 2023-11-25 on EWDS (frozen at
the GloFAS v4.2 release), so reforecast is capped at 2023 while reanalysis
runs through the present.

EWDS request "cost" scales with hyear x hmonth x hday x leadtime_hour, not
with area size or ensemble member count, so requests are chunked in time
(not split per station). The per-request cost cap is not fixed (observed
950 and 500 in the same session), so each request's cost is checked live
against the current limit via estimate_costs(), and requests that exceed
it are recursively bisected on hyear -> hmonth -> hday -> leadtime_hour
until each piece fits.
"""

import logging
import os
import time
from pathlib import Path

import cdsapi
import ocha_stratus as stratus
import pandas as pd
import xarray as xr
import yaml

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"

SCRATCH_DIR = Path(
    os.environ.get(
        "GLOFAS_SCRATCH",
        r"C:\Users\pauni\AppData\Local\Temp\claude\c--Users-pauni-Desktop-Work-OCHA-GitHub-ds-aa-eth-flooding"
        r"\48949c80-5b31-4989-b807-9845baccc6ea\scratchpad\glofas",
    )
)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(SCRATCH_DIR / "fetch_glofas.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fetch_glofas")

VARIABLE = "river_discharge_in_the_last_24_hours"
SYSTEM_VERSION = "version_4_0"
HYDRO_MODEL = "lisflood"
LEADTIME_HOURS = ["24", "48", "72", "96", "120", "144", "168"]  # 1-7 days
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

REANALYSIS_YEARS = [str(y) for y in range(2003, 2026)]  # through 2025
INTERMEDIATE_YEAR = 2026
INTERMEDIATE_MONTHS = [f"{m:02d}" for m in range(1, 8)]  # Jan-Jul available so far

REFORECAST_YEARS = [str(y) for y in range(2003, 2024)]  # EWDS reforecast ends 2023-11-25
REFORECAST_PRODUCT_TYPES = ["control_reforecast", "ensemble_perturbed_reforecast"]

SPLIT_PRIORITY = ["hyear", "hmonth", "hday", "leadtime_hour"]

MASTER_PARQUET_BLOB = f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge.parquet"


def get_client() -> cdsapi.Client:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    return cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key=cfg["key"])


def load_stations() -> pd.DataFrame:
    return stratus.load_csv_from_blob(f"{PROJECT_PREFIX}/raw/glofas/stations.csv", stage=STAGE)


def bounding_box(stations: pd.DataFrame, buffer: float = 0.5) -> list:
    lat_min, lat_max = stations["lat"].min() - buffer, stations["lat"].max() + buffer
    lon_min, lon_max = stations["lon"].min() - buffer, stations["lon"].max() + buffer
    return [lat_max, lon_min, lat_min, lon_max]  # N, W, S, E


def upload_netcdf(local_path: Path, blob_name: str) -> None:
    with open(local_path, "rb") as f:
        stratus.upload_blob_data(
            f.read(), blob_name, stage=STAGE, container_name=CONTAINER,
            content_type="application/x-netcdf",
        )


def extract_points(nc_path: Path, stations: pd.DataFrame, dataset: str, product_type: str) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    frames = []
    for _, st in stations.iterrows():
        sub = ds["dis24"].sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
        df = sub.to_dataframe(name="discharge").reset_index()
        df["station_id"] = st["station_id"]
        df["station_lon"] = st["lon"]
        df["station_lat"] = st["lat"]
        df["dataset"] = dataset
        df["product_type"] = product_type
        frames.append(df)
    ds.close()
    out = pd.concat(frames, ignore_index=True)

    if "forecast_period" in out.columns:
        out["leadtime_days"] = out["forecast_period"].dt.days
        out = out.rename(columns={"forecast_reference_time": "issued_time"})
    else:
        out["leadtime_days"] = 0
        out["issued_time"] = out["valid_time"]

    keep = [c for c in [
        "station_id", "station_lon", "station_lat", "dataset", "product_type",
        "issued_time", "valid_time", "leadtime_days", "number",
        "latitude", "longitude", "discharge",
    ] if c in out.columns]
    return out[keep]


def run_request(client: cdsapi.Client, cds_dataset: str, request: dict, local_path: Path, tries: int = 2) -> bool:
    for attempt in range(1, tries + 1):
        try:
            client.retrieve(cds_dataset, request, str(local_path))
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt}/{tries} failed for {local_path.name}: {e}")
            time.sleep(10)
    log.error(f"Giving up on {local_path.name} after {tries} attempts")
    return False


def submit_with_split(client: cdsapi.Client, coll, cds_dataset: str, request: dict,
                       out_prefix: str, on_downloaded) -> None:
    """Submit a request, recursively bisecting on hyear/hmonth/hday/leadtime_hour
    whenever the live cost estimate exceeds the current cost limit. Calls
    on_downloaded(local_path) immediately after each successful leaf download
    so raw upload + extraction + progress-save happen incrementally."""
    try:
        est = coll.estimate_costs(request)
        cost, limit = est.get("cost"), est.get("limit")
    except Exception as e:
        log.warning(f"Cost estimate failed for {out_prefix} ({e}); submitting as-is")
        cost, limit = None, None

    if cost is not None and limit is not None and cost > limit:
        splittable = [k for k in SPLIT_PRIORITY if isinstance(request.get(k), list) and len(request[k]) > 1]
        if not splittable:
            log.error(f"{out_prefix}: cost {cost} > limit {limit} and cannot split further; skipping")
            return
        key = splittable[0]
        vals = request[key]
        mid = len(vals) // 2
        log.info(f"{out_prefix}: cost {cost} > limit {limit}, splitting {key} ({len(vals)} values) in half")
        for i, part in enumerate((vals[:mid], vals[mid:])):
            sub_request = dict(request)
            sub_request[key] = part
            submit_with_split(client, coll, cds_dataset, sub_request, f"{out_prefix}_{key}{i}", on_downloaded)
        return

    local_path = SCRATCH_DIR / f"{out_prefix}.nc"
    log.info(f"{out_prefix}: requesting (cost={cost}, limit={limit})")
    if run_request(client, cds_dataset, request, local_path):
        on_downloaded(local_path)


def save_progress(all_records: list[pd.DataFrame]) -> None:
    if not all_records:
        return
    combined = pd.concat(all_records, ignore_index=True)
    local_path = SCRATCH_DIR / "glofas_discharge.parquet"
    combined.to_parquet(local_path, index=False)
    stratus.upload_parquet_to_blob(combined, MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
    log.info(f"Progress saved: {len(combined)} rows -> {MASTER_PARQUET_BLOB}")


def _make_on_downloaded(raw_prefix: str, stations: pd.DataFrame, dataset: str,
                         product_type: str, all_records: list):
    def on_downloaded(local_path: Path) -> None:
        blob_name = f"{raw_prefix}/{local_path.stem}.nc"
        upload_netcdf(local_path, blob_name)
        all_records.append(extract_points(local_path, stations, dataset, product_type))
        local_path.unlink(missing_ok=True)
        save_progress(all_records)
    return on_downloaded


def fetch_reanalysis(client: cdsapi.Client, hist_coll, area: list, stations: pd.DataFrame, all_records: list) -> None:
    raw_prefix = f"{PROJECT_PREFIX}/raw/glofas/reanalysis"

    submit_with_split(client, hist_coll, "cems-glofas-historical", {
        "system_version": SYSTEM_VERSION,
        "hydrological_model": HYDRO_MODEL,
        "product_type": "consolidated",
        "variable": VARIABLE,
        "hyear": REANALYSIS_YEARS,
        "hmonth": ALL_MONTHS,
        "hday": ALL_DAYS,
        "data_format": "netcdf",
        "area": area,
    }, "reanalysis_consolidated", _make_on_downloaded(raw_prefix, stations, "reanalysis", "consolidated", all_records))

    # near-real-time months for the current year
    submit_with_split(client, hist_coll, "cems-glofas-historical", {
        "system_version": SYSTEM_VERSION,
        "hydrological_model": HYDRO_MODEL,
        "product_type": "intermediate",
        "variable": VARIABLE,
        "hyear": [str(INTERMEDIATE_YEAR)],
        "hmonth": INTERMEDIATE_MONTHS,
        "hday": ALL_DAYS,
        "data_format": "netcdf",
        "area": area,
    }, f"reanalysis_intermediate_{INTERMEDIATE_YEAR}",
       _make_on_downloaded(raw_prefix, stations, "reanalysis", "intermediate", all_records))


def fetch_reforecast(client: cdsapi.Client, reforecast_coll, area: list, stations: pd.DataFrame, all_records: list) -> None:
    raw_prefix = f"{PROJECT_PREFIX}/raw/glofas/reforecasts"

    for product_type in REFORECAST_PRODUCT_TYPES:
        submit_with_split(client, reforecast_coll, "cems-glofas-reforecast", {
            "system_version": SYSTEM_VERSION,
            "hydrological_model": HYDRO_MODEL,
            "product_type": product_type,
            "variable": VARIABLE,
            "hyear": REFORECAST_YEARS,
            "hmonth": ALL_MONTHS,
            "hday": ALL_DAYS,
            "leadtime_hour": LEADTIME_HOURS,
            "data_format": "netcdf",
            "area": area,
        }, f"reforecast_{product_type}",
           _make_on_downloaded(raw_prefix, stations, "reforecast", product_type, all_records))


def main() -> None:
    client = get_client()
    stations = load_stations()
    area = bounding_box(stations)
    log.info(f"Stations:\n{stations}")
    log.info(f"Bounding box (N, W, S, E): {area}")

    hist_coll = client.client.get_collection("cems-glofas-historical")
    reforecast_coll = client.client.get_collection("cems-glofas-reforecast")

    # Resume support: seed all_records from any previously-saved master parquet
    # so a restart doesn't lose earlier progress, and skip re-fetching a dataset
    # that's already fully present (checked via non-empty raw blob folders).
    all_records: list[pd.DataFrame] = []
    existing = stratus.list_container_blobs(name_starts_with=MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
    if MASTER_PARQUET_BLOB in existing:
        prev = stratus.load_parquet_from_blob(MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
        all_records.append(prev)
        log.info(f"Resumed from existing master parquet: {len(prev)} rows")

    reanalysis_raw = stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/raw/glofas/reanalysis/", stage=STAGE, container_name=CONTAINER,
    )
    reanalysis_files = [b for b in reanalysis_raw if b.endswith(".nc")]
    if reanalysis_files:
        log.info(f"=== Skipping reanalysis fetch: {len(reanalysis_files)} raw files already in blob ===")
    else:
        log.info("=== Starting reanalysis fetch ===")
        fetch_reanalysis(client, hist_coll, area, stations, all_records)

    log.info("=== Starting reforecast fetch ===")
    fetch_reforecast(client, reforecast_coll, area, stations, all_records)

    save_progress(all_records)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
