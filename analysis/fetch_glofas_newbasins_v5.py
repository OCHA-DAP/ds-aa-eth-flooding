"""
Fetch GloFAS v5 reanalysis for the riverine basins OUTSIDE the Somali region
(same boxes as fetch_glofas_newbasins.py, which fetched v4): the lower Omo,
Bilate, Kulfo (Arba Minch), and the Baro and Akobo in Gambella. Used to test
the OND trigger method on v5 reanalysis across all seven river systems.

Two boxes, October to December months only, 2003-2025, consolidated product,
netcdf. Raw box files -> blob raw/glofas/newbasins/ (v5_ prefixed).
Re-runnable: completed chunks are skipped via raw blob names.
"""

import logging
import os
import time
from pathlib import Path

import cdsapi
import ocha_stratus as stratus
import yaml

STAGE = "dev"
CONTAINER = "projects"
PROJECT_PREFIX = "ds-aa-eth-flooding"
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/glofas/newbasins"

SCRATCH_DIR = Path(
    os.environ.get("GLOFAS_SCRATCH", str(Path(__file__).resolve().parent / "scratch_glofas_rp"))
) / "newbasins_v5"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SCRATCH_DIR / "fetch_newbasins_v5.log"), logging.StreamHandler()],
)
log = logging.getLogger("fetch_newbasins_v5")
logging.getLogger("azure").setLevel(logging.WARNING)

# N, W, S, E
BOXES = {
    "gambella": [8.7, 32.9, 6.2, 35.6],   # Baro + Akobo
    "omo_rift": [7.4, 35.6, 4.3, 38.4],   # lower Omo + Bilate + Kulfo
}
YEARS = [str(y) for y in range(2003, 2026)]
OND_MONTHS = ["10", "11", "12"]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
MAX_IN_FLIGHT = 20
POLL_SECONDS = 30


def get_client() -> cdsapi.Client:
    cfg = yaml.safe_load(open(os.path.expanduser("~/.cdsapirc")))
    return cdsapi.Client(
        url="https://ewds.climate.copernicus.eu/api", key=cfg["key"], wait_until_complete=False
    )


def plan(coll, request, prefix):
    try:
        est = coll.estimate_costs(request)
        cost, limit = est.get("cost"), est.get("limit")
    except Exception as e:
        log.warning(f"{prefix}: cost estimate failed ({e}); submitting as-is")
        return [(request, prefix)]
    if cost is not None and limit is not None and cost > limit:
        years = request["year"]
        mid = len(years) // 2
        out = []
        for i, part in enumerate((years[:mid], years[mid:])):
            sub = dict(request)
            sub["year"] = part
            out += plan(coll, sub, f"{prefix}_y{i}")
        return out
    return [(request, prefix)]


def main() -> None:
    client = get_client()
    coll = client.client.get_collection("cems-glofas-historical")
    existing = set(
        stratus.list_container_blobs(name_starts_with=f"{RAW_PREFIX}/", stage=STAGE, container_name=CONTAINER)
    )
    jobs = []
    for name, area in BOXES.items():
        request = {
            "system_version": "version_5_0",
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "average_river_discharge_in_the_last_24_hours",
            "timespan": "time_mean",
            "year": YEARS,
            "month": OND_MONTHS,
            "day": ALL_DAYS,
            "data_format": "netcdf",
            "area": area,
        }
        for req, prefix in plan(coll, request, f"v5_ond_{name}"):
            if f"{RAW_PREFIX}/{prefix}.nc" not in existing:
                jobs.append((req, prefix))
    log.info(f"{len(jobs)} chunks to fetch")

    pending, in_flight = list(jobs), {}
    while pending or in_flight:
        while pending and len(in_flight) < MAX_IN_FLIGHT:
            req, prefix = pending.pop(0)
            try:
                in_flight[prefix] = client.retrieve("cems-glofas-historical", req)
                log.info(f"{prefix}: submitted ({len(pending)} to go)")
            except Exception as e:
                log.error(f"{prefix}: submission failed - {e}")
        for prefix, remote in list(in_flight.items()):
            try:
                remote.update()
                status = remote.status
            except Exception as e:
                log.warning(f"{prefix}: poll failed ({e})")
                continue
            if status == "successful":
                local = SCRATCH_DIR / f"{prefix}.nc"
                remote.download(str(local))
                with open(local, "rb") as f:
                    stratus.upload_blob_data(
                        f.read(), f"{RAW_PREFIX}/{prefix}.nc", stage=STAGE,
                        container_name=CONTAINER, content_type="application/x-netcdf",
                    )
                del in_flight[prefix]
                log.info(f"{prefix}: downloaded + uploaded")
            elif status == "failed":
                log.error(f"{prefix}: FAILED - {remote.get_receipt()}")
                del in_flight[prefix]
        if pending or in_flight:
            time.sleep(POLL_SECONDS)
    log.info("done")


if __name__ == "__main__":
    main()
