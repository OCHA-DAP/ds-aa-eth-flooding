"""
Flooding in the zones along the rivers against rainfall over their catchments,
October to December 1998-2025.

Zones (MoWE flood-prone list): Shabelle (Wabi Shebele); Afder, Liban, Daawa
(Genale Dawa); South Omo (Omo Gibe). For each zone and season: FloodScan SFED
peak (% of zone flooded) and its date, against the catchment's wettest 7-day
rainfall and its date (from basin_rainfall_ond.py). Spearman rank correlation
across the 28 seasons, the lag between rain peak and flood peak, and the daily
lag at which catchment 7-day rainfall correlates best with zone flooding
(Spearman on anomalies from the day-of-year mean; rain days 1 October to 31 December, flood k days later).
Flooding is taken from 1 October to 31 January; rainfall from 1 October to 31 December.

Needs the DB tunnel. Writes processed/rainfall/zone_flood_vs_basin_rain.csv,
zone_flood_correlation.csv, zone_flood_daily_lag.csv.
"""

import numpy as np
import ocha_stratus as stratus
import pandas as pd
from scipy.stats import spearmanr

STAGE = "dev"
P = "ds-aa-eth-flooding/processed/rainfall/"
ZONES = {"ET0506": ("Shabelle", "Wabi Shebele"), "ET0508": ("Afder", "Genale Dawa"), "ET0509": ("Liban", "Genale Dawa"),
         "ET0511": ("Daawa", "Genale Dawa"), "ET0707": ("South Omo", "Omo Gibe")}
OND = [10, 11, 12]
YEARS = range(1998, 2026)

rain = stratus.load_parquet_from_blob(P + "basin_daily_imerg.parquet", stage=STAGE).set_index("date")
d7 = rain.rolling(7, min_periods=7).sum()
eng = stratus.get_engine(stage="prod")
fs = pd.read_sql(f"""select pcode, valid_date, mean from public.floodscan where iso3 = 'ETH' and adm_level = 2
                     and band = 'SFED' and pcode in {tuple(ZONES)} and valid_date between '1998-01-01' and '2026-01-31'""",
                 eng, parse_dates=["valid_date"])
flood = fs.pivot_table(index="valid_date", columns="pcode", values="mean").asfreq("D") * 100
flood.columns = [ZONES[c][0] for c in flood.columns]

rows = []
for pcode, (zone, basin) in ZONES.items():
    for y in YEARS:
        r = d7[basin][(d7.index.year == y) & d7.index.month.isin(OND)].dropna()
        f = flood[zone][(flood.index >= f"{y}-10-01") & (flood.index <= f"{y + 1}-01-31")].dropna()  # flood window Oct-Jan
        rows.append({"zone": zone, "basin": basin, "year": y,
                     "rain7_mm": float(r.max()), "rain_date": r.idxmax().strftime("%Y-%m-%d"),
                     "ond_total_mm": float(rain[basin][(rain.index.year == y) & rain.index.month.isin(OND)].sum()),
                     "flood_peak_pct": float(f.max()), "flood_date": f.idxmax().strftime("%Y-%m-%d"),
                     "lag_days": int((f.idxmax() - r.idxmax()).days)})
res = pd.DataFrame(rows)
g = res.groupby("zone")
res["rank_rain"] = g.rain7_mm.rank(ascending=False, method="first").astype(int)
res["rank_flood"] = g.flood_peak_pct.rank(ascending=False, method="first").astype(int)

corr = []
for zone, gz in res.groupby("zone", sort=False):
    a, b = spearmanr(gz.rain7_mm, gz.flood_peak_pct), spearmanr(gz.ond_total_mm, gz.flood_peak_pct)
    top = gz[gz.rank_rain <= 5]
    corr.append({"zone": zone, "basin": gz.basin.iloc[0], "rho_wettest7": round(a.statistic, 2), "p_wettest7": a.pvalue,
                 "rho_ond_total": round(b.statistic, 2), "p_ond_total": b.pvalue,
                 "top5_rain_also_top5_flood": int((top.rank_flood <= 5).sum()),
                 "median_lag_top5_rain": float(top.lag_days.median())})
corr = pd.DataFrame(corr)
res = res.round({"rain7_mm": 1, "ond_total_mm": 0, "flood_peak_pct": 3})  # rank first, round for output

# daily lag: anomalies from the day-of-year mean (removes the shared seasonal cycle), Spearman
# so a few large seasons do not dominate; lag k = flooding k days after the end of the 7-day window
def anomaly(x: pd.Series) -> pd.Series:
    x = x[(x.index >= "1998-01-01") & (x.index <= "2026-01-31")]
    return x - x.groupby(x.index.dayofyear).transform("mean")


lag_rows = []
for pcode, (zone, basin) in ZONES.items():
    ra, fa = anomaly(d7[basin]), anomaly(flood[zone])
    for k in range(0, 31):
        df = pd.concat([ra.shift(k), fa], axis=1)
        rain_day = df.index - pd.Timedelta(days=k)  # pairs whose rain day is 1 Oct-31 Dec
        df = df[rain_day.month.isin(OND) & (rain_day >= "1998-10-01") & (rain_day <= "2025-12-31")].dropna()
        lag_rows.append({"zone": zone, "lag_days": k, "r": round(float(spearmanr(df.iloc[:, 0], df.iloc[:, 1]).statistic), 3)})
lag = pd.DataFrame(lag_rows)

stratus.upload_csv_to_blob(res, P + "zone_flood_vs_basin_rain.csv", stage=STAGE)
stratus.upload_csv_to_blob(corr, P + "zone_flood_correlation.csv", stage=STAGE)
stratus.upload_csv_to_blob(lag, P + "zone_flood_daily_lag.csv", stage=STAGE)
pd.set_option("display.width", 250)
print(corr.to_string(index=False))
print(res[(res.rank_rain <= 5) | (res.rank_flood <= 5)].sort_values(["zone", "year"]).to_string(index=False))
print(lag.loc[lag.groupby("zone").r.idxmax()].to_string(index=False))
print("saved processed/rainfall/zone_*")
