"""
Fetch GloFAS reanalysis + reforecast at the GloFAS fixed reporting points
inside Ethiopia's Somali region, and push raw + extracted outputs to Azure
Blob via ocha_stratus. Companion to fetch_glofas.py (same structure), for
the official GloFAS reporting points rather than the 5 ad-hoc Shabelle
grid points.

Reporting points were pulled from the GloFAS OWS layer RPG_U
(https://ows.globalfloods.eu/glofas-ows/ows.py, WFS) — the only public
source; the current viewer's API is behind a login. Four fixed points fall
in the wider east-Ethiopia box; the one at (40.45, 5.95) is in Oromia, so
three remain (filtered with the geoBoundaries ETH ADM1 Somali polygon).
Only the Gode point exposes metadata via GetFeatureInfo (station G1904 /
SI004254, Shabelle, drainage 143,466 km2); the other two carry
coordinates only and are named by river reach.

EWDS API note: the cems-glofas-historical dataset was restructured on
2026-07-30 — hyear/hmonth/hday became year/month/day, the variable is now
average_river_discharge_in_the_last_24_hours, and a timespan field was
added. cems-glofas-reforecast still uses the old spelling.

Unlike fetch_glofas.py (which submitted each cost-bisected chunk
sequentially, blocking on the EWDS queue per chunk), this plans all
chunks up front via estimate_costs and keeps a rolling window of them
submitted asynchronously, processing each as it completes — the queue
wait dominates wall time, so overlapping requests is the only real
speed-up available. Completed chunks are skipped on re-run via their raw
blob names.
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
    os.environ.get("GLOFAS_SCRATCH", str(Path(__file__).resolve().parent / "scratch_glofas_rp"))
)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(SCRATCH_DIR / "fetch_glofas_rp.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fetch_glofas_rp")

STATIONS = pd.DataFrame(
    [
        # station_id, name, river, lon, lat
        ("G1904", "Shabelle at Gode (SI004254)", "Shabelle", 44.775, 5.084),
        ("RP_4205_0415", "Genale/Juba at Dolow", "Genale", 42.05, 4.15),
        ("RP_4045_0535", "Genale upper (Somali region W edge)", "Genale", 40.45, 5.35),
    ],
    columns=["station_id", "name", "river", "lon", "lat"],
)

STATIONS_BLOB = f"{PROJECT_PREFIX}/raw/glofas/stations_reporting_points_somali.csv"
MASTER_PARQUET_BLOB = (
    f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge_reporting_points.parquet"
)
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/glofas/reporting_points"

SYSTEM_VERSION = "version_4_0"
HYDRO_MODEL = "lisflood"
LEADTIME_HOURS = ["24", "48", "72", "96", "120", "144", "168"]  # 1-7 days
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

REANALYSIS_YEARS = [str(y) for y in range(2003, 2026)]
INTERMEDIATE_YEAR = 2026
INTERMEDIATE_MONTHS = [f"{m:02d}" for m in range(1, 8)]
REFORECAST_YEARS = [str(y) for y in range(2003, 2024)]  # EWDS reforecast ends 2023-11-25
# flood seasons only (Gu/MAM + Deyr/OND) — reforecast cost multiplies by
# months x lead times, and the all-months version is ~170 EWDS chunks
REFORECAST_MONTHS = ["03", "04", "05", "10", "11", "12"]
REFORECAST_PRODUCT_TYPES = ["control_reforecast", "ensemble_perturbed_reforecast"]

# time-dimension keys eligible for cost bisection, both API spellings
SPLIT_PRIORITY = ["hyear", "year", "hmonth", "month", "hday", "day", "leadtime_hour"]


def get_client() -> cdsapi.Client:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    return cdsapi.Client(
        url="https://ewds.climate.copernicus.eu/api", key=cfg["key"],
        wait_until_complete=False,
    )


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
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    frames = []
    for _, st in stations.iterrows():
        sub = ds[var].sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
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


def plan_chunks(coll, request: dict, out_prefix: str) -> list:
    """Recursively bisect a request until each piece fits the live cost limit.

    Returns a flat list of (request, out_prefix) leaves; makes no
    submissions.
    """
    try:
        est = coll.estimate_costs(request)
        cost, limit = est.get("cost"), est.get("limit")
    except Exception as e:
        log.warning(f"Cost estimate failed for {out_prefix} ({e}); submitting as-is")
        return [(request, out_prefix)]

    if cost is not None and limit is not None and cost > limit:
        splittable = [k for k in SPLIT_PRIORITY if isinstance(request.get(k), list) and len(request[k]) > 1]
        if not splittable:
            log.error(f"{out_prefix}: cost {cost} > limit {limit} and cannot split further; skipping")
            return []
        key = splittable[0]
        vals = request[key]
        mid = len(vals) // 2
        chunks = []
        for i, part in enumerate((vals[:mid], vals[mid:])):
            sub_request = dict(request)
            sub_request[key] = part
            chunks += plan_chunks(coll, sub_request, f"{out_prefix}_{key}{i}")
        return chunks
    return [(request, out_prefix)]


# rolling submission window: enough to keep the per-user processing slots
# busy without tripping EWDS queued-request limits
MAX_IN_FLIGHT = 20
POLL_INTERVAL_SECONDS = 30


def submit_all_and_process(client, jobs: list, existing_blobs: set) -> None:
    """Run every job through EWDS with a rolling async submission window.

    jobs: list of (cds_dataset, request, out_prefix, on_downloaded).
    Chunks whose raw blob already exists are skipped. Each completed
    download is processed immediately via its on_downloaded callback.
    """
    pending = []
    for cds_dataset, request, out_prefix, blob_name, on_downloaded in jobs:
        if blob_name in existing_blobs:
            log.info(f"{out_prefix}: raw blob already present, skipping")
            continue
        pending.append((cds_dataset, request, out_prefix, on_downloaded))

    in_flight = {}  # out_prefix -> (remote, on_downloaded)
    while pending or in_flight:
        while pending and len(in_flight) < MAX_IN_FLIGHT:
            cds_dataset, request, out_prefix, on_downloaded = pending.pop(0)
            try:
                remote = client.retrieve(cds_dataset, request)
                in_flight[out_prefix] = (remote, on_downloaded)
                log.info(f"{out_prefix}: submitted ({len(pending)} still to submit)")
            except Exception as e:
                log.error(f"{out_prefix}: submission failed - {e}")

        for out_prefix, (remote, on_downloaded) in list(in_flight.items()):
            try:
                remote.update()
                status = remote.status
            except Exception as e:
                log.warning(f"{out_prefix}: poll failed ({e}), retrying next cycle")
                continue
            if status == "successful":
                local_path = SCRATCH_DIR / f"{out_prefix}.nc"
                remote.download(str(local_path))
                del in_flight[out_prefix]
                log.info(f"{out_prefix}: downloaded")
                on_downloaded(local_path)
            elif status == "failed":
                log.error(f"{out_prefix}: FAILED - {remote.get_receipt()}")
                del in_flight[out_prefix]

        if pending or in_flight:
            time.sleep(POLL_INTERVAL_SECONDS)


def save_progress(all_records: list) -> None:
    if not all_records:
        return
    combined = pd.concat(all_records, ignore_index=True)
    combined.to_parquet(SCRATCH_DIR / "glofas_discharge_rp.parquet", index=False)
    stratus.upload_parquet_to_blob(combined, MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
    log.info(f"Progress saved: {len(combined)} rows -> {MASTER_PARQUET_BLOB}")


def _make_on_downloaded(raw_prefix: str, stations: pd.DataFrame, dataset: str, product_type: str, all_records: list):
    def on_downloaded(local_path: Path) -> None:
        blob_name = f"{raw_prefix}/{local_path.stem}.nc"
        upload_netcdf(local_path, blob_name)
        all_records.append(extract_points(local_path, stations, dataset, product_type))
        local_path.unlink(missing_ok=True)
        save_progress(all_records)
    return on_downloaded


def build_jobs(hist_coll, reforecast_coll, area, stations, all_records) -> list:
    """Plan every reanalysis + reforecast chunk as an async-submittable job.

    Returns (cds_dataset, request, out_prefix, raw_blob_name, on_downloaded)
    tuples spanning all products, so the whole workload shares one rolling
    submission window.
    """
    jobs = []

    def add(coll, cds_dataset, request, out_prefix, raw_prefix, dataset, product_type):
        cb = _make_on_downloaded(raw_prefix, stations, dataset, product_type, all_records)
        for req, prefix in plan_chunks(coll, request, out_prefix):
            jobs.append((cds_dataset, req, prefix, f"{raw_prefix}/{prefix}.nc", cb))

    ra_prefix = f"{RAW_PREFIX}/reanalysis"
    add(hist_coll, "cems-glofas-historical", {
        "system_version": SYSTEM_VERSION,
        "hydrological_model": HYDRO_MODEL,
        "product_type": "consolidated",
        "variable": "average_river_discharge_in_the_last_24_hours",
        "timespan": "time_mean",
        "year": REANALYSIS_YEARS,
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        "data_format": "netcdf",
        "area": area,
    }, "rp_reanalysis_consolidated", ra_prefix, "reanalysis", "consolidated")

    add(hist_coll, "cems-glofas-historical", {
        "system_version": SYSTEM_VERSION,
        "hydrological_model": HYDRO_MODEL,
        "product_type": "intermediate",
        "variable": "average_river_discharge_in_the_last_24_hours",
        "timespan": "time_mean",
        "year": [str(INTERMEDIATE_YEAR)],
        "month": INTERMEDIATE_MONTHS,
        "day": ALL_DAYS,
        "data_format": "netcdf",
        "area": area,
    }, f"rp_reanalysis_intermediate_{INTERMEDIATE_YEAR}", ra_prefix, "reanalysis", "intermediate")

    rf_prefix = f"{RAW_PREFIX}/reforecasts"
    for product_type in REFORECAST_PRODUCT_TYPES:
        add(reforecast_coll, "cems-glofas-reforecast", {
            "system_version": SYSTEM_VERSION,
            "hydrological_model": HYDRO_MODEL,
            "product_type": product_type,
            "variable": "river_discharge_in_the_last_24_hours",
            "hyear": REFORECAST_YEARS,
            "hmonth": REFORECAST_MONTHS,
            "hday": ALL_DAYS,
            "leadtime_hour": LEADTIME_HOURS,
            "data_format": "netcdf",
            "area": area,
        }, f"rp_reforecast_{product_type}", rf_prefix, "reforecast", product_type)

    return jobs


def main() -> None:
    client = get_client()
    stations = STATIONS
    area = bounding_box(stations)
    log.info(f"Stations:\n{stations}")
    log.info(f"Bounding box (N, W, S, E): {area}")

    stratus.upload_csv_to_blob(stations, STATIONS_BLOB, stage=STAGE, container_name=CONTAINER)
    log.info(f"Stations table -> {STATIONS_BLOB}")

    hist_coll = client.client.get_collection("cems-glofas-historical")
    reforecast_coll = client.client.get_collection("cems-glofas-reforecast")

    all_records: list = []
    existing = stratus.list_container_blobs(name_starts_with=MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
    if MASTER_PARQUET_BLOB in existing:
        prev = stratus.load_parquet_from_blob(MASTER_PARQUET_BLOB, stage=STAGE, container_name=CONTAINER)
        all_records.append(prev)
        log.info(f"Resumed from existing master parquet: {len(prev)} rows")

    log.info("=== Planning chunks (live cost estimates) ===")
    jobs = build_jobs(hist_coll, reforecast_coll, area, stations, all_records)
    log.info(f"=== {len(jobs)} chunks planned; submitting with rolling window of {MAX_IN_FLIGHT} ===")

    existing_raw = set(
        stratus.list_container_blobs(name_starts_with=f"{RAW_PREFIX}/", stage=STAGE, container_name=CONTAINER)
    )
    submit_all_and_process(client, jobs, existing_raw)

    save_progress(all_records)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
