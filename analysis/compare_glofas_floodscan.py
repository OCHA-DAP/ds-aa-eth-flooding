"""
Merge GloFAS reanalysis discharge (per station) with FloodScan flooded-area
(per admin2 zone) for the Shabelle basin stations, using the station -> zone
mapping in ds-aa-eth-flooding/processed/glofas/station_zone_mapping.csv.

Reforecast-based comparison is a follow-up once the full ensemble fetch
finishes; this covers the reanalysis-vs-FloodScan performance check.
"""

import ocha_stratus as stratus
import pandas as pd

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"


def main() -> pd.DataFrame:
    mapping = stratus.load_csv_from_blob(f"{PROJECT_PREFIX}/processed/glofas/station_zone_mapping.csv", stage=STAGE)

    glofas = stratus.load_parquet_from_blob(f"{PROJECT_PREFIX}/processed/glofas/glofas_discharge.parquet", stage=STAGE)
    reanalysis = glofas[glofas["dataset"] == "reanalysis"].copy()
    reanalysis["valid_time"] = pd.to_datetime(reanalysis["valid_time"]).dt.normalize()
    # consolidated/intermediate overlap a little at the boundary; keep the latest value per station/date
    reanalysis = reanalysis.sort_values("valid_time").drop_duplicates(["station_id", "valid_time"], keep="last")

    floodscan = stratus.load_csv_from_blob(f"{PROJECT_PREFIX}/processed/floodscan/floodscan_somali_eth.csv", stage=STAGE)
    floodscan["valid_date"] = pd.to_datetime(floodscan["valid_date"])

    merged = reanalysis.merge(mapping[["station_id", "pcode", "adm2_name"]], on="station_id", how="left")
    merged = merged.merge(
        floodscan[["pcode", "valid_date", "sum"]].rename(columns={"valid_date": "valid_time", "sum": "flooded_extent"}),
        on=["pcode", "valid_time"], how="inner",
    )
    merged = merged[["station_id", "pcode", "adm2_name", "valid_time", "discharge", "flooded_extent"]]

    stratus.upload_parquet_to_blob(merged, f"{PROJECT_PREFIX}/processed/comparison/glofas_reanalysis_floodscan.parquet", stage=STAGE)
    print(f"{len(merged)} rows merged, date range {merged['valid_time'].min()} to {merged['valid_time'].max()}")
    corr = merged.groupby("station_id", group_keys=False).apply(
        lambda g: g["discharge"].corr(g["flooded_extent"], method="spearman"), include_groups=False,
    )
    print("Spearman correlation (discharge vs flooded extent):")
    print(corr)
    return merged


if __name__ == "__main__":
    main()
