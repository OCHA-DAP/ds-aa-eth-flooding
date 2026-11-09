"""
Daily IMERG rainfall over the Wabi Shebele, Genale Dawa and Omo Gibe basins
for the watch page, from the team DB.

Runs as the Databricks job in databricks.yml: GitHub runners cannot reach the
DB, so this job writes the basin series to blob and the page build
(make_dashboard.py, GitHub Actions) reads it from there.

A basin's daily rainfall is the area-weighted mean of the zones it overlaps,
with the same weights as basin_rainfall_ond.py (weight = zone area inside the
basin, processed/rainfall/basin_zone_weights.csv). public.imerg is IMERG Late
Run v7, about two days behind.

Writes processed/rainfall/basin_rainfall_live.parquet (date + one column per
basin, mm/day) for the last DAYS_KEPT days.
"""

from datetime import date, timedelta

import ocha_stratus as stratus
import pandas as pd

STAGE = "dev"
P = "ds-aa-eth-flooding/processed/rainfall/"
BASINS = ["Wabi Shebele", "Genale Dawa", "Omo Gibe"]
DAYS_KEPT = 150  # covers 1 October to 31 December plus the 6 days a 7-day total needs before it

weights = stratus.load_csv_from_blob(P + "basin_zone_weights.csv", stage=STAGE)
pc = tuple(sorted(weights.ADM2_PCODE.unique()))
start = date.today() - timedelta(days=DAYS_KEPT)
imerg = pd.read_sql(f"""select pcode, valid_date, mean from public.imerg where iso3 = 'ETH' and adm_level = 2
                        and pcode in {pc} and valid_date >= '{start}'""",
                    stratus.get_engine(stage="prod"), parse_dates=["valid_date"])
wide = imerg.pivot_table(index="valid_date", columns="pcode", values="mean")

out = {}
for b, g in weights.groupby("basin"):
    w = g.set_index("ADM2_PCODE").area_in_basin_km2.reindex(wide.columns).dropna()
    sub = wide[w.index]
    out[b] = sub.mul(w, axis=1).sum(axis=1, min_count=1) / sub.notna().mul(w, axis=1).sum(axis=1)
daily = pd.DataFrame(out)[BASINS].asfreq("D").round(2)

stratus.upload_parquet_to_blob(daily.reset_index(names="date"), P + "basin_rainfall_live.parquet", stage=STAGE)
print(f"basin rainfall {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d} "
      f"({len(wide.columns)} zones) -> {P}basin_rainfall_live.parquet")
