"""
Possible trigger for the three basins, backtested on the 2003-2025 seasons.

Readiness: GloFAS river flow reaches the river's threshold on the watch
(reached years from processed/comparison/ond_trigger_activations.parquet).
Action: basin 7-day IMERG rainfall, 1 October to 31 December, reaches the
basin's threshold = its 2nd-highest season maximum over 2003-2025 (1-in-12
per basin), so that either stage is met no more than 1 year in 3.
Checked against the river-zone FloodScan series (1 October to 31 January) and
the impact record used on the methods page (EM-DAT, CERF, FloodScan 1-in-3 and
1-in-5 events in the Somali region).

Reads the outputs of basin_rainfall_ond.py. Writes
processed/rainfall/possible_trigger_thresholds.csv, possible_trigger_events.csv,
possible_trigger_years.csv.
"""

import ocha_stratus as stratus
import pandas as pd

STAGE = "dev"
P = "ds-aa-eth-flooding/processed/"
BASINS = ["Wabi Shebele", "Genale Dawa", "Omo Gibe"]
GLOFAS_RIVER = {"Wabi Shebele": "Wabi Shebelle", "Genale Dawa": "Genale Dawa", "Omo Gibe": "Omo"}
YEARS = list(range(2003, 2026))
DATA_DELAY_DAYS = 2  # IMERG reaches the database about two days after each day
ACTION_RANK = 2  # action threshold = the basin's ACTION_RANK-th highest season, 2003-2025

rain = stratus.load_parquet_from_blob(P + "rainfall/basin_daily_imerg.parquet", stage=STAGE).set_index("date")
d7 = rain.rolling(7, min_periods=7).sum()
flood = stratus.load_parquet_from_blob(P + "rainfall/basin_daily_floodscan.parquet", stage=STAGE).set_index("date")
act = stratus.load_parquet_from_blob(P + "comparison/ond_trigger_activations.parquet", stage=STAGE).set_index("year")
imp = stratus.load_parquet_from_blob(P + "comparison/ond_impact_record.parquet", stage=STAGE).set_index("year")

# GloFAS daily flow / threshold per river, as 04_ond_trigger_design (for the date the threshold was first reached)
frames = []
for blob in ["glofas_discharge.parquet", "glofas_discharge_reporting_points.parquet"]:
    df = stratus.load_parquet_from_blob(P + f"glofas/{blob}", stage=STAGE)
    frames.append(df[df.dataset == "reanalysis"][["station_id", "valid_time", "discharge"]])
frames.append(stratus.load_parquet_from_blob(P + "glofas/glofas_discharge_newbasins_ond.parquet",
                                             stage=STAGE)[["station_id", "valid_time", "discharge"]])
gq = pd.concat(frames, ignore_index=True)
gq["date"] = pd.to_datetime(gq.valid_time).dt.normalize() - pd.Timedelta(days=1)
gq = gq.sort_values("valid_time").drop_duplicates(["station_id", "date"], keep="last")
gq = gq[gq.date.dt.month.isin([10, 11, 12]) & gq.date.dt.year.between(2003, 2025)]
lv = stratus.load_csv_from_blob(P + "glofas/ond_trigger_levels.csv", stage=STAGE).set_index("station_id")
gq = gq[gq.station_id.isin(lv.index)]
gq["ratio"] = gq.discharge / gq.station_id.map(lv.threshold_m3s)
gday = gq.pivot_table(index="date", columns=gq.station_id.map(lv.river), values="ratio", aggfunc="max")


def first_on_or_over(x: pd.Series, thr: float):
    hit = x[x >= thr]
    return hit.index[0] if len(hit) else None


thr_rows, events = [], []
for b in BASINS:
    season_max = pd.Series({y: d7[b][f"{y}-10-01":f"{y}-12-31"].max() for y in YEARS})
    thr = float(season_max.nlargest(ACTION_RANK).iloc[-1])
    ranked = season_max.sort_values(ascending=False)
    thr_rows.append({"basin": b, "threshold_mm": round(thr, 1), "next_below_mm": round(float(ranked.iloc[ACTION_RANK]), 1)})
    fpk = {y: flood[b][f"{y}-10-01":f"{y + 1}-01-31"] for y in YEARS}
    fpeak = pd.Series({y: v.max() for y, v in fpk.items()})
    frank = fpeak.rank(ascending=False, method="first").astype(int)
    for y in YEARS:
        action = bool(season_max[y] >= thr)
        ready = bool(act.loc[y, GLOFAS_RIVER[b]])
        if not (action or ready):
            continue
        rc = first_on_or_over(d7[b][f"{y}-10-01":f"{y}-12-31"], thr) if action else None
        # GloFAS reanalysis peaks are compared with the threshold at 3 decimals, as in the reached years
        gc = first_on_or_over(gday[GLOFAS_RIVER[b]][f"{y}-10-01":f"{y}-12-31"].round(3), 1.0) if ready else None
        peak = fpk[y].idxmax()
        events.append({
            "basin": b, "year": y, "readiness": ready, "action": action,
            "glofas_reached": gc.strftime("%Y-%m-%d") if gc is not None else None,
            "rain_reached": rc.strftime("%Y-%m-%d") if rc is not None else None,
            "rain_available": (rc + pd.Timedelta(days=DATA_DELAY_DAYS)).strftime("%Y-%m-%d") if rc is not None else None,
            "flood_peak": peak.strftime("%Y-%m-%d"), "flood_peak_pct": round(float(fpeak[y]), 2), "flood_rank": int(frank[y]),
            "days_glofas_to_flood_peak": int((peak - gc).days) if gc is not None else None,
            "days_rain_available_to_flood_peak": int((peak - rc).days) - DATA_DELAY_DAYS if rc is not None else None,
        })
ev = pd.DataFrame(events)

yr = pd.DataFrame({"year": YEARS})
yr["readiness"] = yr.year.isin(ev[ev.readiness].year)
yr["action"] = yr.year.isin(ev[ev.action].year)
yr["emdat"] = yr.year.map(imp.emdat_events > 0)
yr["cerf"] = yr.year.map(imp.cerf_allocations > 0)
yr["floodscan_rp3_somali"] = yr.year.map(imp.floodscan_rp3_somali)
yr["floodscan_rp5_somali"] = yr.year.map(imp.floodscan_rp5_somali)

for a, b in [(yr.readiness, "readiness"), (yr.action, "action"), (yr.readiness | yr.action, "overall")]:
    n = int(a.sum())
    print(f"{b:9s} {n} seasons, once every {(len(YEARS) + 1) / n:.1f} years: {yr[a].year.tolist()}")
stratus.upload_csv_to_blob(pd.DataFrame(thr_rows), P + "rainfall/possible_trigger_thresholds.csv", stage=STAGE)
stratus.upload_csv_to_blob(ev, P + "rainfall/possible_trigger_events.csv", stage=STAGE)
stratus.upload_csv_to_blob(yr, P + "rainfall/possible_trigger_years.csv", stage=STAGE)
pd.set_option("display.width", 250)
print(pd.DataFrame(thr_rows).to_string(index=False))
print(ev.to_string(index=False))
print("saved processed/rainfall/possible_trigger_*")
