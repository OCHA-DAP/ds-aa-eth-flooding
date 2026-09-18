"""
Build the Somali-region flood monitoring page: station-level only.

Reads from blob:
- processed/glofas/glofas_forecast_latest.parquet   (fetch_glofas_forecast_live.py)
- processed/glofas/glofas_return_periods.parquet    (notebook 01)
- processed/glofas/station_zone_mapping_all.csv
- processed/dashboard/river_cells.json (map river layer, precomputed)

For each station: the GloFAS ensemble-median forecast over the next 10 days
against the station's seasonal 1-in-3 year level (Gu = Mar-Jun, Jul-Sep
highland flows, Deyr = Oct-Dec; each day uses its own season's level). A
station is "reached" when the median is at or above the level on any day.
No trigger logic: station-level information only.

Thresholds are PINNED to the GloFAS version the operational forecast runs
(THRESHOLD_VERSION_PIN, currently v4) and a climatology-coherence check
warns when that stops being true.

Writes analysis/dashboard/eth_flood_dashboard.html (artifact copy) and
docs/index.html (GitHub Pages copy) from template.html (placeholder
__DASHBOARD_DATA__), and uploads the JSON payload + run log to blob.
"""

import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd

STAGE = "dev"
PROJECT_PREFIX = "ds-aa-eth-flooding"
SEASON_MONTHS = {"gu": [3, 4, 5, 6], "kiremt": [7, 8, 9], "deyr": [10, 11, 12]}
RP = 3
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "dashboard"
OUT_DIR.mkdir(exist_ok=True)
RUN_LOG_BLOB = f"{PROJECT_PREFIX}/processed/dashboard/run_log.json"

# The operational forecast and the GloFAS map viewer thresholds are v4-scale
# (verified 2026-08-31: viewer shows RP1.5 ~ 200 m3/s at the Webe Gestro point,
# matching v4 annual RP2 = 236, not v5's 59). Pin v4 so forecast and thresholds
# always share a scale; the climatology matcher below runs as a tripwire and
# the build warns if the operational system stops matching the pin.
THRESHOLD_VERSION_PIN = "v4_0"


def month_season(m: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if m in months:
            return season
    return "jilaal"


def build_map_payload() -> list:
    """Somali-region admin2 outlines, simplified for inline SVG rendering."""
    shp = stratus.load_blob_data("eth_shp.zip", stage=STAGE, container_name="polygon")
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "eth_shp.zip")
    with open(zpath, "wb") as f:
        f.write(shp)
    zipfile.ZipFile(zpath).extractall(tmp)
    adm2 = gpd.read_file(os.path.join(tmp, "eth_adm2.shp"))
    som = adm2[adm2["ADM2_PCODE"].str.startswith("ET05")].copy()
    som["geometry"] = som.geometry.simplify(0.02)
    zones = []
    for _, z in som.iterrows():
        geoms = z.geometry.geoms if z.geometry.geom_type == "MultiPolygon" else [z.geometry]
        rings = [
            [[round(x, 3), round(y, 3)] for x, y in g.exterior.coords]
            for g in geoms
        ]
        zones.append({"pcode": z["ADM2_PCODE"], "name": z["ADM2_EN"], "rings": rings})
    return zones


def load_river_cells() -> list:
    """GloFAS channel cells for the map's river layer (precomputed on blob)."""
    try:
        data = stratus.load_blob_data(
            f"{PROJECT_PREFIX}/processed/dashboard/river_cells.json",
            stage=STAGE, container_name="projects",
        )
        return json.loads(data)["cells"]
    except Exception:
        return []


def update_run_log(issue: pd.Timestamp, stations_payload: list) -> list:
    """Append today's run to the persistent update log (one entry per date,
    newest first, capped at 30). A station is logged when its ensemble-median
    forecast reaches the 1-in-3 year level on any day."""
    try:
        log = json.loads(stratus.load_blob_data(RUN_LOG_BLOB, stage=STAGE, container_name="projects"))
    except Exception:
        log = []
    reached = [
        {"station": st["label"], "level": f"RP{RP}"}
        for st in stations_payload if st["reached"]
    ]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {
        "date": today,
        "issue": issue.strftime("%Y-%m-%d"),
        "checked_at": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "reached": reached,
    }
    log = [e for e in log if e.get("date") != today]
    log.insert(0, entry)
    log = log[:30]
    stratus.upload_blob_data(
        json.dumps(log).encode(), RUN_LOG_BLOB,
        stage=STAGE, container_name="projects", content_type="application/json",
    )
    return log


def load_reanalysis_by_version() -> pd.DataFrame:
    """Daily reanalysis per version | station | date (for climatology matching)."""
    frames = []
    for blob in ["glofas_discharge.parquet", "glofas_discharge_reporting_points.parquet"]:
        df = stratus.load_parquet_from_blob(f"{PROJECT_PREFIX}/processed/glofas/{blob}", stage=STAGE)
        df = df[df["dataset"] == "reanalysis"].copy()
        df["version"] = "v4_0"
        frames.append(df[["version", "station_id", "valid_time", "discharge"]])
    try:
        v5 = stratus.load_parquet_from_blob(
            f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge_v5.parquet", stage=STAGE
        )
        v5["version"] = "v5_0"
        frames.append(v5[["version", "station_id", "valid_time", "discharge"]])
    except Exception:
        pass
    out = pd.concat(frames, ignore_index=True)
    out["valid_time"] = pd.to_datetime(out["valid_time"]).dt.normalize()
    return out


def pick_threshold_version(fc: pd.DataFrame, rean: pd.DataFrame, rivers: pd.Series) -> tuple:
    """Match the operational forecast to the GloFAS version whose climatology it
    follows. The test is COHERENCE, not closeness to 1: a real weather anomaly
    shifts a whole basin by a similar factor, while a version mismatch distorts
    stations by station-specific factors. Ratios are averaged per RIVER first
    (the five Shabelle points are one near-identical series in v4 and must not
    vote five times), then the version with the smallest spread of log-ratios
    across rivers wins."""
    doys = set(pd.to_datetime(fc["valid_time"]).dt.dayofyear)
    window = rean[rean["valid_time"].dt.dayofyear.isin(doys)]
    fc_med = fc.groupby("station_id")["discharge"].median()
    scores = {}
    for version, g in window.groupby("version"):
        rean_med = g.groupby("station_id")["discharge"].median()
        ratio = (fc_med / rean_med).dropna()
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        logr = np.log(ratio).groupby(rivers).mean()  # one vote per river
        scores[version] = float((logr - logr.median()).abs().median())
    best = min(scores, key=scores.get)
    return best, scores


def main() -> None:
    fc = stratus.load_parquet_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/glofas_forecast_latest.parquet", stage=STAGE
    )
    thresholds = stratus.load_parquet_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/glofas_return_periods.parquet", stage=STAGE
    )
    mapping = stratus.load_csv_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/station_zone_mapping_all.csv", stage=STAGE
    )

    fc["issued_time"] = pd.to_datetime(fc["issued_time"])
    fc["valid_time"] = pd.to_datetime(fc["valid_time"])
    issue = fc["issued_time"].max()
    fc = fc[fc["issued_time"] == issue]

    rivers = mapping.set_index("station_id")["river"]
    matched, match_scores = pick_threshold_version(fc, load_reanalysis_by_version(), rivers)
    version = THRESHOLD_VERSION_PIN or matched
    print(f"thresholds: {version} (pinned) | climatology match: {matched} "
          f"(spread per version: { {k: round(v, 2) for k, v in match_scores.items()} })")
    if matched != version:
        print(f"WARNING: forecast climatology now matches {matched}, not the pinned {version}. "
              "The operational GloFAS system may have been upgraded - re-check the map viewer "
              "thresholds and update THRESHOLD_VERSION_PIN.")
    thr = thresholds[thresholds["version"] == version].set_index(
        ["station_id", "season", "rp"]
    )["threshold_gumbel"]

    stations_payload = []
    for _, meta in mapping.iterrows():
        sid = meta["station_id"]
        sub = fc[fc["station_id"] == sid]
        if len(sub) == 0:
            continue
        ens = sub[sub["product_type"] != "control_forecast"]

        leads = []
        for lead, g in ens.groupby("leadtime_days"):
            valid = g["valid_time"].iloc[0]
            season = month_season(valid.month)
            thr_season = season if season != "jilaal" else "annual"
            key = (sid, thr_season, RP)
            level = float(thr.loc[key]) if key in thr.index else None
            median = float(g["discharge"].median())
            leads.append({
                "lead": int(lead),
                "valid": valid.strftime("%Y-%m-%d"),
                "season": season,
                "median": round(median, 1),
                "level": round(level, 1) if level is not None else None,
                "ratio": round(median / level, 3) if level else None,
            })
        leads.sort(key=lambda x: x["lead"])

        with_ratio = [l for l in leads if l["ratio"] is not None]
        peak = max(with_ratio, key=lambda l: l["ratio"]) if with_ratio else None
        stations_payload.append({
            "id": sid,
            "label": meta["label"],
            "river": meta["river"],
            "zone": meta["zone"],
            "lat": round(float(meta["station_lat"]), 3),
            "lon": round(float(meta["station_lon"]), 3),
            "leads": leads,
            "peak": peak["median"] if peak else None,
            "peak_date": peak["valid"] if peak else None,
            "peak_ratio": peak["ratio"] if peak else None,
            "peak_level": peak["level"] if peak else None,
            "reached": bool(peak and peak["ratio"] >= 1),
        })

    payload = {
        "issued": issue.strftime("%Y-%m-%d"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold_version": version,
        "rp": RP,
        "season_now": month_season(datetime.now(timezone.utc).month),
        "map": build_map_payload(),
        "rivers": load_river_cells(),
        "run_log": update_run_log(issue, stations_payload)[:7],
        "stations": stations_payload,
    }

    data_json = json.dumps(payload, allow_nan=False)
    stratus.upload_blob_data(
        data_json.encode(),
        f"{PROJECT_PREFIX}/processed/dashboard/dashboard_data.json",
        stage=STAGE, container_name="projects", content_type="application/json",
    )

    template = (OUT_DIR / "template.html").read_text(encoding="utf-8")
    page = template.replace("__DASHBOARD_DATA__", data_json)
    (OUT_DIR / "eth_flood_dashboard.html").write_text(page, encoding="utf-8")
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(page, encoding="utf-8")
    n_reached = sum(s["reached"] for s in stations_payload)
    print(f"dashboard built: issue {payload['issued']}, thresholds {version}, "
          f"{len(stations_payload)} stations, {n_reached} at 1-in-{RP} -> {docs / 'index.html'}")


if __name__ == "__main__":
    main()
