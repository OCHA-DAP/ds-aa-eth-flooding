"""
Fetch GloFAS v5 reanalysis (cems-glofas-historical, version_5_0) for the box
covering all 8 Somali-region monitoring points, 1980-2026, and extract
channel-snapped series per station.

Why v5: the operational GloFAS forecast (cems-glofas-forecast,
system_version=operational) runs the current system, and in the Somalia
version comparison v4 reanalysis ran 4-5x wet vs observed discharge while
v5 was near-unbiased. Thresholds for a live dashboard must be fitted on
the same system version the live forecast runs, so this fetch provides the
v5 climatology (1980-2025 gives 46 annual maxima vs 23 for the v4 record).

Snapping: v4's discharge is near-flat along the channel while v5 declines
downstream, so independent snapping can pick different cells per version.
Each station is anchored to its v4 channel cell (from the resnap /
stations.csv registries) wherever that cell still carries water in v5;
only dry anchors are re-snapped on the v5 mean field.

Uses the new EWDS API spelling (year/month/day + timespan, post 2026-07-30
restructure). Raw box netcdfs -> raw/glofas/v5_reanalysis/ on blob;
extracted master -> processed/glofas/glofas_discharge_v5.parquet.
Completed chunks are skipped on re-run via their raw blob names.
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
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/glofas/v5_reanalysis"
MASTER_PARQUET_BLOB = f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge_v5.parquet"
SNAPPED_V5_BLOB = f"{PROJECT_PREFIX}/raw/glofas/stations_v5_cells.csv"

SCRATCH_DIR = Path(
    os.environ.get("GLOFAS_SCRATCH", str(Path(__file__).resolve().parent / "scratch_glofas_rp"))
) / "v5"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SCRATCH_DIR / "fetch_v5.log"), logging.StreamHandler()],
)
log = logging.getLogger("fetch_v5")
logging.getLogger("azure").setLevel(logging.WARNING)

YEARS = [str(y) for y in range(1980, 2027)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
AREA = [6.575, 39.95, 3.65, 45.325]  # N, W, S, E: all 8 stations + 0.5deg
DRY_FLOOR_M3S = 10.0

MAX_IN_FLIGHT = 20
POLL_INTERVAL_SECONDS = 30
SPLIT_PRIORITY = ["year", "month", "day"]


def get_client() -> cdsapi.Client:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    return cdsapi.Client(
        url="https://ewds.climate.copernicus.eu/api", key=cfg["key"],
        wait_until_complete=False,
    )


def load_station_cells() -> pd.DataFrame:
    """v4 channel cells for all 8 stations: grid registry + snapped reporting points."""
    grid = stratus.load_csv_from_blob(
        f"{PROJECT_PREFIX}/raw/glofas/stations.csv", stage=STAGE
    ).rename(columns={"lon": "anchor_lon", "lat": "anchor_lat"})
    snapped = stratus.load_csv_from_blob(
        f"{PROJECT_PREFIX}/raw/glofas/stations_reporting_points_somali_snapped.csv", stage=STAGE
    )
    snapped = snapped.rename(columns={"snapped_lon": "anchor_lon", "snapped_lat": "anchor_lat"})
    cols = ["station_id", "anchor_lon", "anchor_lat"]
    return pd.concat([grid[cols], snapped[cols]], ignore_index=True)


def plan_chunks(coll, request: dict, out_prefix: str) -> list:
    try:
        est = coll.estimate_costs(request)
        cost, limit = est.get("cost"), est.get("limit")
    except Exception as e:
        log.warning(f"Cost estimate failed for {out_prefix} ({e}); submitting as-is")
        return [(request, out_prefix)]
    if cost is not None and limit is not None and cost > limit:
        splittable = [k for k in SPLIT_PRIORITY if isinstance(request.get(k), list) and len(request[k]) > 1]
        if not splittable:
            log.error(f"{out_prefix}: cost {cost} > limit {limit}, cannot split; skipping")
            return []
        key = splittable[0]
        vals = request[key]
        mid = len(vals) // 2
        out = []
        for i, part in enumerate((vals[:mid], vals[mid:])):
            sub = dict(request)
            sub[key] = part
            out += plan_chunks(coll, sub, f"{out_prefix}_{key}{i}")
        return out
    return [(request, out_prefix)]


def submit_all(client, jobs: list) -> list:
    """Rolling-window submission; returns local paths of downloaded chunks."""
    downloaded = []
    pending = list(jobs)
    in_flight = {}
    while pending or in_flight:
        while pending and len(in_flight) < MAX_IN_FLIGHT:
            request, out_prefix = pending.pop(0)
            try:
                remote = client.retrieve("cems-glofas-historical", request)
                in_flight[out_prefix] = remote
                log.info(f"{out_prefix}: submitted ({len(pending)} to go)")
            except Exception as e:
                log.error(f"{out_prefix}: submission failed - {e}")
        for out_prefix, remote in list(in_flight.items()):
            try:
                remote.update()
                status = remote.status
            except Exception as e:
                log.warning(f"{out_prefix}: poll failed ({e})")
                continue
            if status == "successful":
                local = SCRATCH_DIR / f"{out_prefix}.nc"
                remote.download(str(local))
                del in_flight[out_prefix]
                blob_name = f"{RAW_PREFIX}/{out_prefix}.nc"
                with open(local, "rb") as f:
                    stratus.upload_blob_data(
                        f.read(), blob_name, stage=STAGE, container_name=CONTAINER,
                        content_type="application/x-netcdf",
                    )
                downloaded.append(local)
                log.info(f"{out_prefix}: downloaded + uploaded")
            elif status == "failed":
                log.error(f"{out_prefix}: FAILED - {remote.get_receipt()}")
                del in_flight[out_prefix]
        if pending or in_flight:
            time.sleep(POLL_INTERVAL_SECONDS)
    return downloaded


def discharge_var(ds: xr.Dataset) -> str:
    return next(v for v in ds.data_vars if "dis" in v.lower())


def snap_v5(paths: list, anchors: pd.DataFrame) -> pd.DataFrame:
    sums, count = None, 0
    for p in paths:
        ds = xr.open_dataset(p)
        da = ds[discharge_var(ds)]
        tdim = next(d for d in da.dims if "time" in d)
        s = da.sum(tdim)
        sums = s if sums is None else sums + s
        count += da.sizes[tdim]
        ds.close()
    mean_field = sums / count

    rows = []
    for _, st in anchors.iterrows():
        at_anchor = mean_field.sel(
            latitude=st["anchor_lat"], longitude=st["anchor_lon"], method="nearest"
        )
        if float(at_anchor) >= DRY_FLOOR_M3S:
            lon, lat, mean = float(at_anchor["longitude"]), float(at_anchor["latitude"]), float(at_anchor)
            resnapped = False
        else:
            box = mean_field.sel(
                latitude=slice(st["anchor_lat"] + 0.15, st["anchor_lat"] - 0.15),
                longitude=slice(st["anchor_lon"] - 0.15, st["anchor_lon"] + 0.15),
            )
            flat = box.stack(cell=("latitude", "longitude"))
            best = flat.isel(cell=int(flat.argmax("cell")))
            lon, lat, mean = float(best["longitude"]), float(best["latitude"]), float(best)
            resnapped = True
        rows.append({
            "station_id": st["station_id"], "v5_lon": lon, "v5_lat": lat,
            "v5_cell_mean_m3s": mean, "resnapped": resnapped,
        })
        log.info(f"{st['station_id']}: v5 cell ({lat}, {lon}), mean {mean:.0f} m3/s, resnapped={resnapped}")
    return pd.DataFrame(rows)


def extract(paths: list, cells: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for p in paths:
        ds = xr.open_dataset(p)
        var = discharge_var(ds)
        for _, st in cells.iterrows():
            sub = ds[var].sel(latitude=st["v5_lat"], longitude=st["v5_lon"], method="nearest")
            df = sub.to_dataframe(name="discharge").reset_index()
            df["station_id"] = st["station_id"]
            frames.append(df)
        ds.close()
    out = pd.concat(frames, ignore_index=True)
    tcol = next(c for c in out.columns if "time" in c)
    out = out.rename(columns={tcol: "valid_time"})
    keep = [c for c in ["station_id", "valid_time", "latitude", "longitude", "discharge"] if c in out.columns]
    return out[keep]


def main() -> None:
    client = get_client()
    coll = client.client.get_collection("cems-glofas-historical")
    anchors = load_station_cells()
    log.info(f"anchors:\n{anchors}")

    request = {
        "system_version": "version_5_0",
        "hydrological_model": "lisflood",
        "product_type": "consolidated",
        "variable": "average_river_discharge_in_the_last_24_hours",
        "timespan": "time_mean",
        "year": YEARS,
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        "data_format": "netcdf",
        "area": AREA,
    }
    jobs = plan_chunks(coll, request, "v5_reanalysis")
    existing = set(
        stratus.list_container_blobs(name_starts_with=f"{RAW_PREFIX}/", stage=STAGE, container_name=CONTAINER)
    )
    todo = [(r, p) for r, p in jobs if f"{RAW_PREFIX}/{p}.nc" not in existing]
    log.info(f"{len(jobs)} chunks planned, {len(todo)} to fetch")
    submit_all(client, todo)

    paths = sorted(SCRATCH_DIR.glob("v5_reanalysis*.nc"))
    # pull down any chunks fetched in earlier runs but missing locally
    for b in existing:
        name = Path(b).name
        if name.endswith(".nc") and not (SCRATCH_DIR / name).exists():
            (SCRATCH_DIR / name).write_bytes(
                stratus.load_blob_data(b, stage=STAGE, container_name=CONTAINER)
            )
    paths = sorted(SCRATCH_DIR.glob("v5_reanalysis*.nc"))
    log.info(f"{len(paths)} local chunks for extraction")

    cells = snap_v5(paths, anchors)
    stratus.upload_csv_to_blob(cells, SNAPPED_V5_BLOB, stage=STAGE)
    combined = extract(paths, cells)
    stratus.upload_parquet_to_blob(combined, MASTER_PARQUET_BLOB, stage=STAGE)
    log.info(f"{len(combined)} rows -> {MASTER_PARQUET_BLOB}")


if __name__ == "__main__":
    main()
