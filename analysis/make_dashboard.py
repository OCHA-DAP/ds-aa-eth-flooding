"""
Build the GloFAS forecast dashboard for the Somali-region stations.

Reads from blob:
- processed/glofas/glofas_forecast_latest.parquet   (fetch_glofas_forecast_live.py)
- processed/glofas/glofas_return_periods.parquet    (notebook 01; newest version wins)
- processed/glofas/station_zone_mapping_all.csv
- processed/comparison/reforecast_trigger_skill.parquet (notebook 03)

Computes, per station and lead: ensemble quantiles, control run, and the
probability of exceeding the seasonal RP2 / RP3 / RP5 thresholds (each valid
date uses its own season's threshold: Gu = Mar-Jun | Kiremt = Jul-Sep |
Deyr = Oct-Dec). Historic trigger skill (2003-2023 v4 reforecast vs FloodScan
RP3 events, averaged over leads because skill is lead-flat) is attached per
station | season as context.

Writes analysis/dashboard/eth_flood_dashboard.html from template.html
(placeholder __DASHBOARD_DATA__) and uploads the JSON payload to blob for
the record.
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
RP_SHOW = [2, 3, 5]
SEASON_MONTHS = {"gu": [3, 4, 5, 6], "kiremt": [7, 8, 9], "deyr": [10, 11, 12]}
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "dashboard"
OUT_DIR.mkdir(exist_ok=True)

# Deyr monitoring mode: thresholds and trigger status use the Deyr fit only,
# and only forecast days falling in Deyr months count toward the trigger
# (Kiremt flows routinely exceed the Deyr thresholds on the Shabelle, so an
# ungated Deyr threshold would false-fire through September).
SEASON_FOCUS = "deyr"

# proposed trigger legs for Deyr 2026 (from notebook 03: v4 reforecast
# 2003-2023 vs FloodScan RP3 events; skill flat across leads 1-7)
TRIGGER_LEGS = [
    {
        "id": "dolow_rp5",
        "station_id": "RP_4205_0415",
        "name": "Genale/Juba at Dolow",
        "rule": "≥50% of members over Deyr RP5, any lead",
        "rp": 5, "prob_gate": 0.5,
        "grade": "Action-grade",
        "skill": "POD 0.29 · FAR 0.00 (validated, zero false alarms 2003-2023)",
    },
    {
        "id": "genale_upper_rp2",
        "station_id": "RP_4045_0535",
        "name": "Genale upper",
        "rule": "≥50% of members over Deyr RP2, any lead",
        "rp": 2, "prob_gate": 0.5,
        "grade": "Watch-grade",
        "skill": "POD 0.67 · FAR 0.55 (sensitive; use as heads-up, not release)",
    },
    {
        "id": "gode_rp3",
        "station_id": "G1904",
        "name": "Shabelle at Gode",
        "rule": "≥50% of members over Deyr RP3, any lead",
        "rp": 3, "prob_gate": 0.5,
        "grade": "Pending v5",
        "skill": "v4 forecast blind in Deyr (POD 0.00) — awaiting v5 rescore before this leg is usable",
    },
]


def month_season(m: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if m in months:
            return season
    return "jilaal"


def build_map_payload(validation_pcodes: set) -> list:
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
        zones.append({
            "pcode": z["ADM2_PCODE"],
            "name": z["ADM2_EN"],
            "validation": z["ADM2_PCODE"] in validation_pcodes,
            "rings": rings,
        })
    return zones


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
    skill = stratus.load_parquet_from_blob(
        f"{PROJECT_PREFIX}/processed/comparison/reforecast_trigger_skill.parquet", stage=STAGE
    )
    fs_events = stratus.load_parquet_from_blob(
        f"{PROJECT_PREFIX}/processed/floodscan/floodscan_event_catalogue.parquet", stage=STAGE
    )
    fs_events["peak_date"] = pd.to_datetime(fs_events["peak_date"])

    version = sorted(thresholds["version"].unique())[-1]
    thr = thresholds[thresholds["version"] == version].set_index(
        ["station_id", "season", "rp"]
    )["threshold_gumbel"]

    fc["issued_time"] = pd.to_datetime(fc["issued_time"])
    fc["valid_time"] = pd.to_datetime(fc["valid_time"])
    issue = fc["issued_time"].max()
    fc = fc[fc["issued_time"] == issue]

    stations_payload = []
    for _, meta in mapping.iterrows():
        sid = meta["station_id"]
        sub = fc[fc["station_id"] == sid]
        if len(sub) == 0:
            continue
        ens = sub[sub["product_type"] != "control_forecast"]
        ctl = sub[sub["product_type"] == "control_forecast"].set_index("leadtime_days")["discharge"]

        leads = []
        for lead, g in ens.groupby("leadtime_days"):
            valid = g["valid_time"].iloc[0]
            season = month_season(valid.month)
            # threshold basis: the focus season's fit in monitoring mode,
            # otherwise each valid date's own season
            thr_season = SEASON_FOCUS or season
            in_season = season == thr_season
            probs, thr_vals = {}, {}
            for rp in RP_SHOW:
                key = (sid, thr_season, rp)
                t = float(thr.loc[key]) if key in thr.index else None
                thr_vals[f"rp{rp}"] = t
                probs[f"rp{rp}"] = (
                    float((g["discharge"] >= t).mean()) if t is not None else None
                )
            q = g["discharge"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            leads.append({
                "lead": int(lead),
                "valid": valid.strftime("%Y-%m-%d"),
                "season": season,
                "in_season": in_season,
                "p10": round(float(q.loc[0.1]), 1),
                "p25": round(float(q.loc[0.25]), 1),
                "median": round(float(q.loc[0.5]), 1),
                "p75": round(float(q.loc[0.75]), 1),
                "p90": round(float(q.loc[0.9]), 1),
                "control": round(float(ctl.get(lead, np.nan)), 1) if lead in ctl.index else None,
                "n_members": int(g["discharge"].count()),
                "prob": probs,
                "thresholds": thr_vals,
            })
        leads.sort(key=lambda x: x["lead"])

        # historic trigger skill, lead-averaged (skill is flat across leads)
        skill_seasons = [SEASON_FOCUS] if SEASON_FOCUS else ["gu", "deyr"]
        sk = skill[(skill["station_id"] == sid) & (skill["season"].isin(skill_seasons))]
        sk = sk[(sk["prob"].isin(["0.5", "control"])) & (sk["rp"].isin([2, 3, 5]))]
        skill_notes = [
            {
                "season": season, "rp": int(rp),
                "pod": round(float(g["pod"].mean()), 2),
                "far": round(float(g["far"].mean()), 2),
                "kind": "ensemble p50" if (g["prob"] == "0.5").any() else "control run",
            }
            for (season, rp), g in sk.groupby(["season", "rp"])
            if g["pod"].notna().any()
        ]

        counted = [l for l in leads if l["in_season"]]
        max_prob_rp3 = max(((l["prob"]["rp3"] or 0) for l in counted), default=0)
        max_prob_rp5 = max(((l["prob"]["rp5"] or 0) for l in counted), default=0)
        max_prob_rp2 = max(((l["prob"]["rp2"] or 0) for l in counted), default=0)
        if not counted:
            status = "preseason"
        elif max_prob_rp5 >= 0.3:
            status = "critical"
        elif max_prob_rp3 >= 0.3:
            status = "serious"
        elif max_prob_rp2 >= 0.3:
            status = "warning"
        else:
            status = "good"

        focus = SEASON_FOCUS or "deyr"
        focus_thr = {
            f"rp{rp}": (round(float(thr.loc[(sid, focus, rp)]), 0) if (sid, focus, rp) in thr.index else None)
            for rp in RP_SHOW
        }
        ev = fs_events[(fs_events["pcode"] == meta["pcode"]) & (fs_events["season"] == focus)]
        event_years = {
            f"rp{r}": sorted(int(y) for y in ev[ev["fs_rp"] == r]["peak_date"].dt.year.unique())
            for r in [2, 3]
        }

        stations_payload.append({
            "id": sid,
            "label": meta["label"],
            "river": meta["river"],
            "zone": meta["zone"],
            "lat": round(float(meta["station_lat"]), 3),
            "lon": round(float(meta["station_lon"]), 3),
            "status": status,
            "max_prob": {"rp2": max_prob_rp2, "rp3": max_prob_rp3, "rp5": max_prob_rp5},
            "deyr_thresholds": focus_thr,
            "deyr_event_years": event_years,
            "leads": leads,
            "skill": skill_notes,
        })

    legs_payload = []
    if SEASON_FOCUS:
        by_id = {s["id"]: s for s in stations_payload}
        for leg in TRIGGER_LEGS:
            st = by_id.get(leg["station_id"])
            counted = [l for l in st["leads"] if l["in_season"]] if st else []
            cur = max(((l["prob"][f"rp{leg['rp']}"] or 0) for l in counted), default=None)
            legs_payload.append({
                **{k: leg[k] for k in ["id", "station_id", "name", "rule", "grade", "skill"]},
                "current_prob": cur,
                "reached": (cur is not None and cur >= leg["prob_gate"]),
                "in_window": bool(counted),
            })

    payload = {
        "issued": issue.strftime("%Y-%m-%d"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold_version": version,
        "threshold_fit": "Gumbel on 2003-2025 seasonal maxima",
        "season_focus": SEASON_FOCUS,
        "season_label": "Deyr 2026 (Oct-Dec)" if SEASON_FOCUS == "deyr" else None,
        "legs": legs_payload,
        "map": build_map_payload(set(mapping["pcode"])),
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
    # artifact copy: fragment (the artifact publisher adds the document skeleton)
    (OUT_DIR / "eth_flood_dashboard.html").write_text(page, encoding="utf-8")
    # GitHub Pages copy: full document at docs/index.html
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Shabelle & Genale Flood Watch</title>\n"
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"dashboard built: issue {payload['issued']}, thresholds {version}, "
          f"{len(stations_payload)} stations -> {OUT_DIR / 'eth_flood_dashboard.html'} + {docs / 'index.html'}")


if __name__ == "__main__":
    main()
