"""
Re-extract the GloFAS reporting-point series from the raw netcdfs on blob,
with channel snapping.

fetch_glofas_reporting_points.py extracted each station with a plain
nearest-cell lookup, and all three official reporting points sit just off
GloFAS's own 0.05deg river network: the nearest cells are dry (Gode reads
0 m3/s everywhere, Dolow ~1 m3/s). Same trap as Jowhar in the Somalia
work. This script rebuilds the master parquet by snapping each station to
the maximum-mean-discharge cell within a search window (0.15deg, widened
to 0.25deg if the best cell still looks dry), using the consolidated
reanalysis as the discharge climatology, then re-extracting reanalysis and
reforecast at the snapped cells.

Overwrites processed/glofas/glofas_discharge_reporting_points.parquet
(the previous contents were unusable) and writes the snapped registry to
raw/glofas/stations_reporting_points_somali_snapped.csv. Raw netcdfs are
untouched.
"""

import logging
import os
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import xarray as xr

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/glofas/reporting_points"
MASTER_PARQUET_BLOB = (
    f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge_reporting_points.parquet"
)
SNAPPED_STATIONS_BLOB = (
    f"{PROJECT_PREFIX}/raw/glofas/stations_reporting_points_somali_snapped.csv"
)

SCRATCH_DIR = Path(
    os.environ.get("GLOFAS_SCRATCH", str(Path(__file__).resolve().parent / "scratch_glofas_rp"))
) / "resnap"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("resnap")
logging.getLogger("azure").setLevel(logging.WARNING)

STATIONS = pd.DataFrame(
    [
        ("G1904", "Shabelle at Gode (SI004254)", "Shabelle", 44.775, 5.084),
        ("RP_4205_0415", "Genale/Juba at Dolow", "Genale", 42.05, 4.15),
        ("RP_4045_0535", "Genale upper (Somali region W edge)", "Genale", 40.45, 5.35),
    ],
    columns=["station_id", "name", "river", "lon", "lat"],
)

WINDOW_DEG = 0.15
WIDE_WINDOW_DEG = 0.25
DRY_FLOOR_M3S = 10.0  # mean discharge below this means we still missed the channel


def download_raw(subdir: str) -> list:
    blobs = [
        b
        for b in stratus.list_container_blobs(
            name_starts_with=f"{RAW_PREFIX}/{subdir}/", stage=STAGE, container_name=CONTAINER
        )
        if b.endswith(".nc")
    ]
    paths = []
    for b in blobs:
        local = SCRATCH_DIR / Path(b).name
        if not local.exists():
            data = stratus.load_blob_data(b, stage=STAGE, container_name=CONTAINER)
            local.write_bytes(data)
            log.info(f"downloaded {b} ({len(data) / 1e6:.1f} MB)")
        paths.append(local)
    return paths


def discharge_var(ds: xr.Dataset) -> str:
    return next(v for v in ds.data_vars if "dis" in v.lower())


def build_mean_field(reanalysis_paths: list) -> xr.DataArray:
    """Time-mean discharge per cell over the consolidated reanalysis."""
    sums, counts = None, 0
    for p in reanalysis_paths:
        ds = xr.open_dataset(p)
        da = ds[discharge_var(ds)]
        s = da.sum("valid_time")
        n = da.sizes["valid_time"]
        sums = s if sums is None else sums + s
        counts += n
        ds.close()
    return sums / counts


def snap_stations(mean_field: xr.DataArray) -> pd.DataFrame:
    rows = []
    for _, st in STATIONS.iterrows():
        chosen = None
        for window in (WINDOW_DEG, WIDE_WINDOW_DEG):
            box = mean_field.sel(
                latitude=slice(st["lat"] + window, st["lat"] - window),
                longitude=slice(st["lon"] - window, st["lon"] + window),
            )
            flat = box.stack(cell=("latitude", "longitude"))
            best = flat.isel(cell=int(flat.argmax("cell")))
            chosen = {
                "station_id": st["station_id"],
                "name": st["name"],
                "river": st["river"],
                "lon": st["lon"],
                "lat": st["lat"],
                "snapped_lon": float(best["longitude"]),
                "snapped_lat": float(best["latitude"]),
                "cell_mean_m3s": float(best),
                "window_deg": window,
            }
            if chosen["cell_mean_m3s"] >= DRY_FLOOR_M3S:
                break
            log.warning(
                f"{st['station_id']}: best cell in {window}deg window still dry "
                f"({chosen['cell_mean_m3s']:.1f} m3/s), widening"
            )
        rows.append(chosen)
        log.info(
            f"{st['station_id']}: ({st['lat']}, {st['lon']}) -> "
            f"({chosen['snapped_lat']}, {chosen['snapped_lon']}), "
            f"mean {chosen['cell_mean_m3s']:.0f} m3/s, window {chosen['window_deg']}deg"
        )
    return pd.DataFrame(rows)


def extract_points(nc_path: Path, snapped: pd.DataFrame, dataset: str, product_type: str) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    var = discharge_var(ds)
    frames = []
    for _, st in snapped.iterrows():
        sub = ds[var].sel(
            latitude=st["snapped_lat"], longitude=st["snapped_lon"], method="nearest"
        )
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


def classify(path: Path) -> tuple:
    stem = path.stem
    if "reanalysis" in stem:
        return "reanalysis", ("intermediate" if "intermediate" in stem else "consolidated")
    return "reforecast", (
        "control_reforecast" if "control" in stem else "ensemble_perturbed_reforecast"
    )


def main() -> None:
    rean_paths = download_raw("reanalysis")
    rf_paths = download_raw("reforecasts")
    log.info(f"{len(rean_paths)} reanalysis + {len(rf_paths)} reforecast files")

    consolidated = [p for p in rean_paths if "consolidated" in p.stem]
    mean_field = build_mean_field(consolidated)
    snapped = snap_stations(mean_field)
    stratus.upload_csv_to_blob(snapped, SNAPPED_STATIONS_BLOB, stage=STAGE)
    log.info(f"snapped registry -> {SNAPPED_STATIONS_BLOB}")

    records = []
    for p in rean_paths + rf_paths:
        dataset, product_type = classify(p)
        records.append(extract_points(p, snapped, dataset, product_type))
        log.info(f"extracted {p.name}")
    combined = pd.concat(records, ignore_index=True)
    stratus.upload_parquet_to_blob(combined, MASTER_PARQUET_BLOB, stage=STAGE)
    log.info(f"{len(combined)} rows -> {MASTER_PARQUET_BLOB}")


if __name__ == "__main__":
    main()
