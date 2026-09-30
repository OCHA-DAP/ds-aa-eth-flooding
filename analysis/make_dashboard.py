"""
Build the riverine flood monitoring page for the EDRMC OND zones.

The trigger (notebook 04, working group 29 Sep 2026): reached when any of the
seven river systems (Wabi Shebelle | Genale Dawa | Omo | Bilate | Gamo lakes |
Baro | Akobo) has a station at or over its threshold on a forecast day in
October to December. Thresholds start at each river's 3rd-largest normalised
OND seasonal peak on GloFAS v4 reanalysis 2003-2025 and are lowered where that
catches further seasons already inside the overall activation years (adjusted
30 Sep 2026; a 1-in-8 to 1-in-6 event per river), chosen so the overall
any-river frequency is 1-in-3.0 (8 activations in 23 years: 2008, 2011, 2013,
2017, 2019, 2023, 2024, 2025).

Reads from blob:
- processed/glofas/glofas_forecast_latest.parquet   (fetch_glofas_forecast_live.py)
- processed/glofas/ond_trigger_levels.csv           (notebook 04: station thresholds)
- processed/glofas/station_zone_mapping_all.csv
- processed/dashboard/river_cells.json

The page checks the GloFAS ensemble median at leads 1 to 10 days; days outside
October to December do not count. The version pin below guards the v4 scale.

Writes analysis/dashboard/eth_flood_dashboard.html and docs/index.html from
template.html (placeholder __DASHBOARD_DATA__), and uploads the JSON payload
and run log to blob.
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
OND = [10, 11, 12]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "dashboard"
OUT_DIR.mkdir(exist_ok=True)
RUN_LOG_BLOB = f"{PROJECT_PREFIX}/processed/dashboard/run_log.json"

RIVERS = ["Wabi Shebelle", "Genale Dawa", "Omo", "Bilate", "Gamo lakes", "Baro", "Akobo"]
# EDRMC riverine zones (ADM2_EN) -> the river systems that put them at risk
ZONE_RIVERS = {
    "Shabelle": ["Wabi Shebelle"],
    "Afder": ["Genale Dawa"], "Liban": ["Genale Dawa"], "Daawa": ["Genale Dawa"],
    "South Omo": ["Omo"],
    "Sidama": ["Bilate"], "West Guji": ["Bilate"],
    "Gamo": ["Gamo lakes"],
    "Itang Special woreda": ["Baro"], "Nuwer": ["Baro"],
    "Agnewak": ["Baro", "Akobo"],
}

# The operational forecast and the GloFAS map viewer thresholds are v4-scale
# (verified 2026-08-31: viewer RP1.5 ~ 200 m3/s at the Webe Gestro point =
# v4 annual RP2 236, not v5's 59). The climatology matcher below runs as a
# tripwire and the build warns if the operational system stops matching.
THRESHOLD_VERSION_PIN = "v4_0"


def build_map_payload() -> dict:
    """Region outlines for context plus the riverine adm2 zones."""
    shp = stratus.load_blob_data("eth_shp.zip", stage=STAGE, container_name="polygon")
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "eth_shp.zip")
    with open(zpath, "wb") as f:
        f.write(shp)
    zipfile.ZipFile(zpath).extractall(tmp)

    def rings_of(geom):
        geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        return [[[round(x, 3), round(y, 3)] for x, y in g.exterior.coords] for g in geoms]

    adm1 = gpd.read_file(os.path.join(tmp, "eth_adm1.shp"))
    adm1["geometry"] = adm1.geometry.simplify(0.03)
    regions = [{"name": r["ADM1_EN"], "rings": rings_of(r.geometry)} for _, r in adm1.iterrows()]

    adm2 = gpd.read_file(os.path.join(tmp, "eth_adm2.shp"))
    riverine = adm2[adm2["ADM2_EN"].isin(ZONE_RIVERS)].copy()
    riverine["geometry"] = riverine.geometry.simplify(0.02)
    zones = [
        {"name": z["ADM2_EN"], "pcode": z["ADM2_PCODE"],
         "rivers": ZONE_RIVERS[z["ADM2_EN"]], "rings": rings_of(z.geometry)}
        for _, z in riverine.iterrows()
    ]
    return {"regions": regions, "zones": zones}


def load_river_cells() -> list:
    try:
        data = stratus.load_blob_data(
            f"{PROJECT_PREFIX}/processed/dashboard/river_cells.json",
            stage=STAGE, container_name="projects",
        )
        return json.loads(data)["cells"]
    except Exception:
        return []


def update_run_log(issue: pd.Timestamp, rivers_status: dict) -> None:
    try:
        log = json.loads(stratus.load_blob_data(RUN_LOG_BLOB, stage=STAGE, container_name="projects"))
    except Exception:
        log = []
    reached = [{"station": r, "level": "OND trigger level"} for r, v in rivers_status.items() if v["reached"]]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {
        "date": today,
        "issue": issue.strftime("%Y-%m-%d"),
        "checked_at": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "reached": reached,
    }
    log = [e for e in log if e.get("date") != today]
    log.insert(0, entry)
    stratus.upload_blob_data(
        json.dumps(log[:60]).encode(), RUN_LOG_BLOB,
        stage=STAGE, container_name="projects", content_type="application/json",
    )


def load_reanalysis_by_version() -> pd.DataFrame:
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
    """Coherence test: a real weather anomaly shifts a whole basin by a similar
    factor, a version mismatch distorts stations by station-specific factors.
    Log-ratios are averaged per river before scoring the spread."""
    doys = set(pd.to_datetime(fc["valid_time"]).dt.dayofyear)
    window = rean[rean["valid_time"].dt.dayofyear.isin(doys)]
    fc_med = fc.groupby("station_id")["discharge"].median()
    scores = {}
    for version, g in window.groupby("version"):
        rean_med = g.groupby("station_id")["discharge"].median()
        ratio = (fc_med / rean_med).dropna()
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        logr = np.log(ratio).groupby(rivers).mean()
        scores[version] = float((logr - logr.median()).abs().median())
    best = min(scores, key=scores.get)
    return best, scores


def main() -> None:
    fc = stratus.load_parquet_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/glofas_forecast_latest.parquet", stage=STAGE
    )
    levels = stratus.load_csv_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/ond_trigger_levels.csv", stage=STAGE
    ).set_index("station_id")
    mapping = stratus.load_csv_from_blob(
        f"{PROJECT_PREFIX}/processed/glofas/station_zone_mapping_all.csv", stage=STAGE
    )

    fc["issued_time"] = pd.to_datetime(fc["issued_time"])
    fc["valid_time"] = pd.to_datetime(fc["valid_time"])
    issue = fc["issued_time"].max()
    fc = fc[fc["issued_time"] == issue]

    rivers_series = mapping.set_index("station_id")["river"]
    matched, match_scores = pick_threshold_version(fc, load_reanalysis_by_version(), rivers_series)
    version = THRESHOLD_VERSION_PIN or matched
    print(f"thresholds: {version} (pinned) | climatology match: {matched} "
          f"(spread per version: { {k: round(v, 2) for k, v in match_scores.items()} })")
    if matched != version:
        print(f"WARNING: forecast climatology now matches {matched}, not the pinned {version}. "
              "The operational GloFAS system may have been upgraded - re-check the map viewer "
              "thresholds and update THRESHOLD_VERSION_PIN.")

    stations_payload = []
    for _, meta in mapping.iterrows():
        sid = meta["station_id"]
        if sid not in levels.index:
            continue
        sub = fc[fc["station_id"] == sid]
        if len(sub) == 0:
            continue
        ens = sub[sub["product_type"] != "control_forecast"]
        level = float(levels.loc[sid, "threshold_m3s"])
        river = levels.loc[sid, "river"]

        leads = []
        for lead, g in ens.groupby("leadtime_days"):
            valid = g["valid_time"].iloc[0]
            median = float(g["discharge"].median())
            leads.append({
                "lead": int(lead),
                "valid": valid.strftime("%Y-%m-%d"),
                "in_ond": valid.month in OND,
                "median": round(median, 1),
                "level": round(level, 1),
                "ratio": round(median / level, 3),
            })
        leads.sort(key=lambda x: x["lead"])
        counting = [l for l in leads if l["in_ond"]]
        peak = max(counting, key=lambda l: l["ratio"]) if counting else None
        stations_payload.append({
            "id": sid,
            "label": meta["label"],
            "river": river,
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

    rivers_status = {}
    for river in RIVERS:
        sts = [s for s in stations_payload if s["river"] == river]
        rivers_status[river] = {
            "reached": any(s["reached"] for s in sts),
            "in_window": any(l["in_ond"] for s in sts for l in s["leads"]),
            "n_stations": len(sts),
            "stations_over": [s["id"] for s in sts if s["reached"]],
        }
    trigger_reached = any(r["reached"] for r in rivers_status.values())

    payload = {
        "issued": issue.strftime("%Y-%m-%d"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold_version": version,
        "trigger_reached": trigger_reached,
        "rivers_status": rivers_status,
        "map": build_map_payload(),
        "rivers": load_river_cells(),
        "stations": stations_payload,
    }
    update_run_log(issue, rivers_status)

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
    (docs / "index.html").write_text(page, encoding="utf-8")
    n_reached = [r for r, v in rivers_status.items() if v["reached"]]
    print(f"dashboard built: issue {payload['issued']}, {len(stations_payload)} stations, "
          f"rivers reached: {n_reached or 'none'} -> {docs / 'index.html'}")


if __name__ == "__main__":
    main()
