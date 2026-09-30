"""
Extract v5 OND reanalysis series for the stations outside the v5 Somali-box
master parquet: the five new-basin stations (from the raw v5_ond_* box files
fetched by fetch_glofas_newbasins_v5.py) plus dawa_1 (its cell sits inside the
Somali legacy box, so it comes from the raw v5_reanalysis box files).

Stations are extracted at their v4 channel cells (the design registry,
processed/glofas/station_zone_mapping_all.csv). For each new-basin cell the
script prints a channel check on the v5 mean field: the cell's mean discharge
against the largest mean within 0.125 degrees, so a cell that falls off the
v5 river network is visible before the series is used.

Output: processed/glofas/glofas_discharge_newbasins_ond_v5.parquet
(station_id, valid_time, discharge), OND 2003-2025.
"""

import io
from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
import xarray as xr

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"
SCRATCH = Path(__file__).resolve().parent / "scratch_glofas_rp"
OUT_BLOB = f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge_newbasins_ond_v5.parquet"

BOX_OF = {"omo_1": "omo_rift", "bilate_1": "omo_rift", "gamo_1": "omo_rift",
          "baro_1": "gambella", "akobo_1": "gambella"}


def open_box(prefix: str, folder: str) -> xr.Dataset:
    paths = sorted((SCRATCH / folder).glob(f"{prefix}*.nc"))
    missing = not paths
    if missing:
        (SCRATCH / folder).mkdir(parents=True, exist_ok=True)
        for blob in stratus.list_container_blobs(
            name_starts_with=f"{PROJECT_PREFIX}/raw/glofas/", stage=STAGE, container_name=CONTAINER
        ):
            name = blob.split("/")[-1]
            if name.startswith(prefix) and name.endswith(".nc"):
                data = stratus.load_blob_data(blob, stage=STAGE, container_name=CONTAINER)
                (SCRATCH / folder / name).write_bytes(
                    data.getvalue() if isinstance(data, io.BytesIO) else data
                )
        paths = sorted((SCRATCH / folder).glob(f"{prefix}*.nc"))
    ds = xr.open_mfdataset(paths, combine="by_coords")
    var = next(v for v in ds.data_vars if "dis" in v.lower())
    return ds[var]


mapping = stratus.load_csv_from_blob(
    f"{PROJECT_PREFIX}/processed/glofas/station_zone_mapping_all.csv", stage=STAGE
).set_index("station_id")

frames = []

for box in ["gambella", "omo_rift"]:
    da = open_box(f"v5_ond_{box}", "newbasins_v5")
    mean_field = da.mean("valid_time").compute()
    for sid, b in BOX_OF.items():
        if b != box:
            continue
        lon, lat = mapping.loc[sid, ["station_lon", "station_lat"]]
        cell = da.sel(latitude=lat, longitude=lon, method="nearest")
        s = cell.to_series()
        frames.append(pd.DataFrame({
            "station_id": sid, "valid_time": s.index.get_level_values("valid_time")
            if isinstance(s.index, pd.MultiIndex) else s.index, "discharge": s.values,
        }))
        cell_mean = float(mean_field.sel(latitude=lat, longitude=lon, method="nearest"))
        win = mean_field.sel(latitude=slice(lat + 0.125, lat - 0.125),
                             longitude=slice(lon - 0.125, lon + 0.125))
        if win.latitude.size == 0:
            win = mean_field.sel(latitude=slice(lat - 0.125, lat + 0.125),
                                 longitude=slice(lon - 0.125, lon + 0.125))
        print(f"{sid}: v5 cell mean {cell_mean:.1f} m3/s | window max mean "
              f"{float(win.max()):.1f} | cell/windowmax {cell_mean / float(win.max()):.2f}")

# dawa_1 from the Somali-box v5 raw files (full-year record; keep OND 2003-2025)
da = open_box("v5_reanalysis", "v5")
lon, lat = mapping.loc["dawa_1", ["station_lon", "station_lat"]]
mean_field = da.mean("valid_time").compute()
cell_mean = float(mean_field.sel(latitude=lat, longitude=lon, method="nearest"))
win = mean_field.sel(latitude=slice(lat + 0.125, lat - 0.125),
                     longitude=slice(lon - 0.125, lon + 0.125))
if win.latitude.size == 0:
    win = mean_field.sel(latitude=slice(lat - 0.125, lat + 0.125),
                         longitude=slice(lon - 0.125, lon + 0.125))
print(f"dawa_1: v5 cell mean {cell_mean:.1f} m3/s | window max mean "
      f"{float(win.max()):.1f} | cell/windowmax {cell_mean / float(win.max()):.2f}")
s = da.sel(latitude=lat, longitude=lon, method="nearest").to_series()
frames.append(pd.DataFrame({
    "station_id": "dawa_1", "valid_time": s.index.get_level_values("valid_time")
    if isinstance(s.index, pd.MultiIndex) else s.index, "discharge": s.values,
}))

out = pd.concat(frames, ignore_index=True).dropna(subset=["discharge"])
out["valid_time"] = pd.to_datetime(out["valid_time"])
d = out["valid_time"] - pd.Timedelta(days=1)
out = out[d.dt.month.isin([10, 11, 12]) & d.dt.year.between(2003, 2025)]
print(f"\n{len(out)} rows | {out['station_id'].nunique()} stations | "
      f"{out['valid_time'].min().date()} to {out['valid_time'].max().date()}")
stratus.upload_parquet_to_blob(out, OUT_BLOB, stage=STAGE)
print(f"saved {OUT_BLOB}")
