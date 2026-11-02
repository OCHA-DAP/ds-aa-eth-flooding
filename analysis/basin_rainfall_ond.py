"""
Wettest October to December weeks over the Wabi Shebele, Genale Dawa and
Omo Gibe basins, 1998-2025, from the team DB.

Rainfall: public.imerg (IMERG daily, zone level, prod DB); public.era5
(monthly) as a second record for the season totals. Flooding: public.floodscan
(SFED band, daily flooded fraction per zone) over the zones along each river
(FLOOD_ZONES), 1 October to 31 January, for the most flooded seasons. The DB holds Ethiopia
down to zones (adm2), so a basin's daily rainfall is the area-weighted mean
of the zones it overlaps (weight = zone area inside the basin).

Per basin and year: the wettest 7-day total between 1 October and 31
December, its date, the season total, and the number of days whose 7-day
total is at or over that month's threshold (P90 of the month's 7-day totals
for October, P99 for November and December, over 1998-2025).

Needs the DB tunnel (see ds-knowledge-base-internal local-db-access).
Basins: processed/basins/eth_rainfall_basins.geojson. Writes to blob:
processed/rainfall/basin_daily_imerg.parquet, basin_ond_years.csv,
basin_ond_thresholds.csv, basin_zone_weights.csv.
"""

import io
import os
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import ocha_stratus as stratus
import pandas as pd

STAGE = "dev"
P = "ds-aa-eth-flooding/"
BASINS = ["Wabi Shebele", "Genale Dawa", "Omo Gibe"]
YEARS = list(range(1998, 2026))
OND = [10, 11, 12]
PCT = {10: 90, 11: 99, 12: 99}
FLOOD_ZONES = {"Wabi Shebele": ["ET0506"],                          # Shabelle
               "Genale Dawa": ["ET0508", "ET0509", "ET0511"],        # Afder, Liban, Daawa
               "Omo Gibe": ["ET0707"]}                               # South Omo

basins = gpd.read_file(stratus.load_blob_data(P + "processed/basins/eth_rainfall_basins.geojson",
                                              stage=STAGE, container_name="projects")).to_crs(32637)
raw = stratus.load_blob_data("eth_shp.zip", stage=STAGE, container_name="polygon")
tmp = tempfile.mkdtemp()
zipfile.ZipFile(io.BytesIO(raw.getvalue() if hasattr(raw, "getvalue") else raw)).extractall(tmp)
zones = gpd.read_file(os.path.join(tmp, "eth_adm2.shp")).to_crs(32637)[["ADM2_PCODE", "ADM2_EN", "geometry"]]
ov = gpd.overlay(zones, basins[["basin", "geometry"]], how="intersection")
ov["area_in_basin_km2"] = ov.area / 1e6
ov = ov[ov.area_in_basin_km2 > 1]
ov["share_of_zone"] = (ov.area / zones.set_index("ADM2_PCODE").area.reindex(ov.ADM2_PCODE).values).round(3)
weights = ov[["basin", "ADM2_PCODE", "ADM2_EN", "area_in_basin_km2", "share_of_zone"]].round(1)

eng = stratus.get_engine(stage="prod")
pc = tuple(sorted(weights.ADM2_PCODE.unique()))
imerg = pd.read_sql(f"""select pcode, valid_date, mean from public.imerg where iso3 = 'ETH' and adm_level = 2
                        and pcode in {pc} and valid_date between '1998-01-01' and '2026-01-31'""",
                    eng, parse_dates=["valid_date"])
era5 = pd.read_sql(f"""select pcode, valid_date, mean from public.era5 where iso3 = 'ETH' and adm_level = 2
                       and pcode in {pc} and valid_date between '1998-01-01' and '2025-12-31'""",
                   eng, parse_dates=["valid_date"])
fs = pd.read_sql(f"""select pcode, valid_date, mean from public.floodscan where iso3 = 'ETH' and adm_level = 2
                     and band = 'SFED' and pcode in {pc} and valid_date between '1998-01-01' and '2026-01-31'""",
                 eng, parse_dates=["valid_date"])


def basin_mean(wide: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for b, g in weights.groupby("basin"):
        w = g.set_index("ADM2_PCODE").area_in_basin_km2.reindex(wide.columns).dropna()
        sub = wide[w.index]
        out[b] = sub.mul(w, axis=1).sum(axis=1, min_count=1) / sub.notna().mul(w, axis=1).sum(axis=1)
    return pd.DataFrame(out)[BASINS]


daily = basin_mean(imerg.pivot_table(index="valid_date", columns="pcode", values="mean")).asfreq("D")
era_m = basin_mean(era5.pivot_table(index="valid_date", columns="pcode", values="mean"))
d7 = daily.rolling(7, min_periods=7).sum()
# flooding is measured in the zones along each river (MoWE flood-prone zones), weighted by full zone area:
# % of those zones flooded
zone_km2 = zones.set_index("ADM2_PCODE").area / 1e6
fs_wide = fs.pivot_table(index="valid_date", columns="pcode", values="mean")
flood = pd.DataFrame({b: fs_wide[pcs].mul(zone_km2[pcs], axis=1).sum(axis=1, min_count=1) / zone_km2[pcs].sum()
                      for b, pcs in FLOOD_ZONES.items()})[BASINS] * 100

rows, thr_rows = [], []
for b in BASINS:
    x = d7[b]
    thr = {m: float(np.percentile(x[(x.index.month == m) & x.index.year.isin(YEARS)].dropna(), p)) for m, p in PCT.items()}
    for m, p in PCT.items():
        thr_rows.append({"basin": b, "month": m, "percentile": p, "mm_7day": round(thr[m], 1)})
    era_ond = era_m[b][era_m.index.month.isin(OND)]
    era_tot = era_ond.groupby(era_ond.index.year).sum()
    for y in YEARS:
        sel = x[(x.index.year == y) & x.index.month.isin(OND)].dropna()
        over = sel[sel >= sel.index.month.map(thr).values]
        fl = flood[b][(flood.index >= f"{y}-10-01") & (flood.index <= f"{y + 1}-01-31")].dropna()  # flood window Oct-Jan
        rows.append({"basin": b, "year": y, "max_7day_mm": float(sel.max()),
                     "date_of_max": sel.idxmax().strftime("%Y-%m-%d"), "days_over": int(len(over)),
                     "ond_total_mm": float(daily[b][(daily.index.year == y) & daily.index.month.isin(OND)].sum()),
                     "era5_ond_total": float(era_tot.get(y, np.nan)),
                     "flood_peak_pct": float(fl.max()), "date_flood_peak": fl.idxmax().strftime("%Y-%m-%d"),
                     "flood_mean_pct": float(fl.mean())})
res = pd.DataFrame(rows)
g = res.groupby("basin")
res["rank_flood"] = g.flood_peak_pct.rank(ascending=False, method="first").astype(int)
res["rank_7day"] = g.max_7day_mm.rank(ascending=False, method="first").astype(int)
res["rank_total"] = g.ond_total_mm.rank(ascending=False, method="first").astype(int)
res["rank_total_era5"] = g.era5_ond_total.rank(ascending=False, method="first").astype(int)
res = res.drop(columns="era5_ond_total").round({"max_7day_mm": 1, "ond_total_mm": 0, "flood_peak_pct": 3, "flood_mean_pct": 3})  # rank first, round for output

# GloFAS (v4 reanalysis, the watch's OND record 2003-2025), as in 04_ond_trigger_design: station season
# peak / station median season peak, max over the river's stations, divided by the river's threshold
# ratio -> season peak relative to the threshold. Reached years are read from the watch's own table.
GLOFAS_RIVER = {"Wabi Shebele": "Wabi Shebelle", "Genale Dawa": "Genale Dawa", "Omo Gibe": "Omo"}
frames = []
for blob in ["glofas_discharge.parquet", "glofas_discharge_reporting_points.parquet"]:
    df = stratus.load_parquet_from_blob(P + f"processed/glofas/{blob}", stage=STAGE)
    frames.append(df[df.dataset == "reanalysis"][["station_id", "valid_time", "discharge"]])
frames.append(stratus.load_parquet_from_blob(P + "processed/glofas/glofas_discharge_newbasins_ond.parquet",
                                             stage=STAGE)[["station_id", "valid_time", "discharge"]])
gq = pd.concat(frames, ignore_index=True)
gq["date"] = pd.to_datetime(gq.valid_time).dt.normalize() - pd.Timedelta(days=1)
gq = gq.sort_values("valid_time").drop_duplicates(["station_id", "date"], keep="last")
gq = gq[gq.date.dt.month.isin(OND) & gq.date.dt.year.between(2003, 2025)]
lv = stratus.load_csv_from_blob(P + "processed/glofas/ond_trigger_levels.csv", stage=STAGE)
smax = gq.pivot_table(index=gq.date.dt.year, columns="station_id", values="discharge", aggfunc="max")
norm = smax / smax.median()
river_of = lv.set_index("station_id").river
ratio_star = (lv.threshold_m3s / lv.median_ond_max_m3s).groupby(lv.river).mean()
act = stratus.load_parquet_from_blob(P + "processed/comparison/ond_trigger_activations.parquet", stage=STAGE).set_index("year")
gl = []
for b, r in GLOFAS_RIVER.items():
    rel = norm[[s for s in norm.columns if river_of.get(s) == r]].max(axis=1) / ratio_star[r]
    gl.append(pd.DataFrame({"basin": b, "year": rel.index, "glofas_ratio": rel.values,
                            "glofas_reached": act[r].reindex(rel.index).values}))
    mism = set(rel.index[rel >= 0.999]) ^ set(act.index[act[r]])
    print(f"GloFAS {b}: reached {sorted(act.index[act[r]])}" + (f" | ratio check differs for {sorted(mism)}" if mism else ""))
gpk = pd.concat(gl)
gpk["glofas_rank"] = gpk.groupby("basin").glofas_ratio.rank(ascending=False, method="first").astype(int)
res = res.merge(gpk[["basin", "year", "glofas_ratio", "glofas_rank", "glofas_reached"]].round(3), on=["basin", "year"], how="left")

stratus.upload_parquet_to_blob(daily.reset_index(names="date"), P + "processed/rainfall/basin_daily_imerg.parquet", stage=STAGE)
stratus.upload_parquet_to_blob(flood.reset_index(names="date"), P + "processed/rainfall/basin_daily_floodscan.parquet", stage=STAGE)
stratus.upload_csv_to_blob(res, P + "processed/rainfall/basin_ond_years.csv", stage=STAGE)
stratus.upload_csv_to_blob(pd.DataFrame(thr_rows), P + "processed/rainfall/basin_ond_thresholds.csv", stage=STAGE)
stratus.upload_csv_to_blob(weights, P + "processed/rainfall/basin_zone_weights.csv", stage=STAGE)
pd.set_option("display.width", 200)
print(res[res.rank_7day <= 5].sort_values(["basin", "rank_7day"]).to_string(index=False))
print(res[res.rank_flood <= 5].sort_values(["basin", "rank_flood"]).to_string(index=False))
print("saved processed/rainfall/*")
