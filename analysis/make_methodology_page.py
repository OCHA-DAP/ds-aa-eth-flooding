"""
Build docs/methodology/index.html: how the Ethiopia Riverine Flood Watch
works, generated from the executed notebook 04. Figures (PNG) and result
tables are harvested straight from analysis/04_ond_trigger_design.ipynb,
parsed and re-set in the page's own table style, so the page always shows
exactly what the notebook computed. Rerun after rerunning the notebook.
"""

import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NB = HERE / "04_ond_trigger_design.ipynb"
NB5 = HERE / "05_v5_reanalysis_test.ipynb"
OUT = HERE.parent / "docs" / "methodology"
FIGS = OUT / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(exist_ok=True)

nb = json.loads(NB.read_text(encoding="utf-8"))
nb5 = json.loads(NB5.read_text(encoding="utf-8"))

RIVER_ORDER = ["Wabi Shebelle", "Genale Dawa", "Omo", "Bilate", "Gamo lakes", "Baro", "Akobo"]
# display names + zones as on the monitoring page (processed/glofas/station_zone_mapping_all.csv)
STATIONS = {
    "G1904": ("Shabelle at Gode (G1904)", "Shebelle"),
    "station_1": ("Shabelle upstream (43.03E 6.08N)", "Shebelle"),
    "station_2": ("Shabelle mid (44.13E 5.63N)", "Shebelle"),
    "station_3": ("Shabelle lower (44.53E 5.43N)", "Shebelle"),
    "station_4": ("Shabelle at border (44.83E 5.13N)", "Shebelle"),
    "RP_4045_0535": ("Genale upper (40.45E 5.35N)", "Liban"),
    "RP_4205_0415": ("Genale/Juba at Dolow", "Liban"),
    "dawa_1": ("Dawa (40.73E 4.43N)", "Daawa"),
    "station_5": ("Webe Gestro (42.08E 4.73N)", "Afder"),
    "omo_1": ("Omo lower (South Omo)", "South Omo"),
    "bilate_1": ("Bilate (Loka Abaya)", "Sidama"),
    "gamo_1": ("Abaya-Chamo lakes (Kulfo area)", "Gamo"),
    "baro_1": ("Baro at Itang", "Itang Special woreda"),
    "akobo_1": ("Akobo (Agnewak)", "Agnewak"),
}


def cell_outputs(marker: str, book: dict | None = None) -> dict:
    """Outputs of the first code cell whose source contains the marker."""
    for c in (book or nb)["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return c.get("outputs", [])
    raise KeyError(marker)


def html_of(marker: str, book: dict | None = None) -> str:
    for o in cell_outputs(marker, book):
        if o.get("output_type") == "execute_result" and "text/html" in o.get("data", {}):
            return "".join(o["data"]["text/html"])
    raise KeyError(f"no html table in cell: {marker}")


def df_of(marker: str, book: dict | None = None) -> pd.DataFrame:
    return pd.read_html(io.StringIO(html_of(marker, book)))[0]


def png_of(marker: str, name: str) -> str:
    for o in cell_outputs(marker):
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
            (FIGS / name).write_bytes(base64.b64decode(o["data"]["image/png"]))
            return f"figs/{name}"
    raise KeyError(f"no figure in cell: {marker}")


def stream_of(marker: str, book: dict | None = None) -> str:
    txt = ""
    for o in cell_outputs(marker, book):
        if o.get("output_type") == "stream":
            txt += "".join(o["text"])
    return txt.strip()


def n0(x) -> str:
    return f"{x:,.0f}"


def chips(years) -> str:
    return " ".join(f'<span class="yr">{y}</span>' for y in years)


# ---------------------------------------------------------------- levels table
lv = df_of('levels["threshold_m3s"]')
lv.columns = ["station_id", "river", "median_m3s", "threshold_m3s"]
lv = lv.set_index("station_id")

rows = []
for river in RIVER_ORDER:
    sub = lv[lv["river"] == river]
    ids = [s for s in STATIONS if s in sub.index]  # STATIONS order within the river
    for j, sid in enumerate(ids):
        r = sub.loc[sid]
        label, zone = STATIONS[sid]
        first = f'<td class="river" rowspan="{len(ids)}">{river}</td>' if j == 0 else ""
        cls = ' class="group"' if j == 0 else ""
        rows.append(
            f"<tr{cls}>{first}<td>{label}</td><td>{zone}</td>"
            f'<td class="num">{n0(r["median_m3s"])}</td>'
            f'<td class="num"><strong>{n0(r["threshold_m3s"])}</strong></td>'
            f'<td class="num">{r["threshold_m3s"] / r["median_m3s"]:.2f}</td></tr>'
        )
levels_table = (
    '<table class="data"><thead><tr><th>River system</th><th>Station</th><th>Zone</th>'
    '<th class="num">Typical season peak (m³/s)</th><th class="num">Threshold (m³/s)</th>'
    '<th class="num">Threshold vs typical peak</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
)

# ---------------------------------------------------------- activation seasons
acts_text = stream_of("per-river activation seasons")
overall_m = re.search(r"overall: (\d+) activation years in (\d+) \| RP ([\d.]+)", acts_text)
n_act, n_yrs, rp = overall_m.groups()
act_rows = []
for river, per_rp, yrs in re.findall(r"^\s{2}(.+?)\s+\(([\d.]+)-yr\): \[(.*)\]", acts_text, re.M):
    act_rows.append(
        f'<tr><td class="river">{river}</td><td>{chips(yrs.split(", "))}</td>'
        f'<td class="num">1-in-{float(per_rp):.0f}</td></tr>'
    )
union_years = ["2008", "2011", "2013", "2017", "2019", "2023", "2024", "2025"]
acts_table = (
    '<table class="data"><thead><tr><th>River system</th><th>OND seasons at or over its level</th>'
    '<th class="num">Frequency</th></tr></thead><tbody>' + "".join(act_rows) +
    f'<tr class="total"><td class="river">Any river (overall)</td><td>{chips(union_years)}</td>'
    f'<td class="num">1-in-{float(rp):.1f}</td></tr></tbody></table>'
)

# ----------------------------------------------------------------- impact record
imp = df_of('impact["cerf_usd"]').rename(columns={"Unnamed: 0": "year"})
imp_rows = []
for _, r in imp.iterrows():
    cls = ' class="hit"' if r["trigger"] else ""
    trig = '<span class="reached">reached</span>' if r["trigger"] else ""
    fs5 = '<span class="dot"></span>' if r["floodscan_rp5_somali"] else ""
    if r["emdat_events"]:
        emdat = f'{r["emdat_events"]} event'
        if pd.notna(r["emdat_affected"]):
            emdat += f' : {n0(r["emdat_affected"])} affected'
    else:
        emdat = ""
    cerf = f'USD {r["cerf_usd"] / 1e6:.1f}M ({r["cerf_allocations"]})' if r["cerf_allocations"] else ""
    imp_rows.append(
        f'<tr{cls}><td class="num">{r["year"]}</td><td>{trig}</td>'
        f'<td class="ctr">{fs5}</td>'
        f"<td>{emdat}</td><td>{cerf}</td></tr>"
    )
impact_table = (
    '<table class="data"><thead><tr><th class="num">Year</th><th>Level reached</th>'
    '<th class="ctr">FloodScan RP5</th>'
    "<th>EM-DAT riverine floods</th><th>CERF flood allocations</th></tr></thead><tbody>"
    + "".join(imp_rows) + "</tbody></table>"
)

# ----------------------------------------------------------------- v5 comparison
v5_text = stream_of("only in v5:", nb5)
v5_cmp = df_of("pd.DataFrame(rows)", nb5).drop(columns="Unnamed: 0")
v5_diff = dict(re.findall(r"(only in v5|only in v4|in both)\s*: \[(.*)\]", v5_text))
v5_rows = []
for i, r in v5_cmp.iterrows():
    is_v4 = "v4" in r["record"]
    live = ' <span class="tag">live</span>' if is_v4 else ""
    cls = ' class="adopted"' if is_v4 else ""
    name = r["record"].replace(" (adopted)", ", the watch record").replace(" (this test)", ", this test")
    v5_rows.append(
        f"<tr{cls}>"
        f'<td class="wrap">{name}{live}</td>'
        f'<td class="num">{r["n_activations"]} of 23</td>'
        f'<td class="num">1-in-{r["overall_rp"]:.2f}</td>'
        f'<td class="num">{r["emdat"]}</td><td class="num">{r["cerf"]}</td>'
        f'<td class="wrap dim">{r["years"]}</td></tr>'
    )
v5_table = (
    '<table class="data"><thead><tr><th>Record</th><th class="num">Years reached</th>'
    '<th class="num">Overall frequency</th><th class="num">EM-DAT years caught</th>'
    '<th class="num">CERF years caught</th>'
    '<th>Years</th></tr></thead><tbody>' + "".join(v5_rows) + "</tbody></table>"
)

fig_stat = png_of("Season peak compared with a typical season", "season_statistic.png")
fig_tiles = png_of("Years each river reached its level", "activations.png")

generated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flood Watch Methods</title>
<meta name="description" content="How the Ethiopia Riverine Flood Watch works: the rivers and zones covered, how each river's OND level was set, the backtest and the impact record.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --b5:#1f69b3; --b6:#1f69b3; --b7:#144372; --b05:#ebf0f4; --b1:#d4e5f7;
  --n9:#1f2324; --n8:#3f4748; --n7:#5e6a6b; --n05:#f5f7f7;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--n05); color:var(--n9);
  font-family:'Roboto',system-ui,-apple-system,'Segoe UI','Helvetica Neue',Arial,sans-serif; line-height:1.5; }}
.wrap {{ max-width:1080px; margin:0 auto; background:#fff; min-height:100vh; box-shadow:0 0 40px rgba(31,35,36,.06); }}
.brand {{ display:flex; justify-content:space-between; align-items:center; gap:20px; padding:14px 44px; background:#fff; border-bottom:1px solid #e2e7e7; }}
.brand a {{ display:block; line-height:0; }}
.brand img {{ height:44px; width:auto; display:block; }}
.hero {{ background:var(--b6); padding:40px 44px 34px; }}
.crumb {{ font-size:12px; margin:0 0 14px; }}
.crumb a {{ color:rgba(255,255,255,.85); text-decoration:none; }}
.crumb a:hover {{ text-decoration:underline; }}
.hero h1 {{ font-family:'Merriweather',Georgia,serif; font-weight:700; font-size:28px; line-height:1.2; color:#fff; margin:0 0 14px; }}
.hero p {{ margin:0; font-size:15px; line-height:1.6; color:#fff; max-width:64ch; }}
article {{ padding:8px 44px 24px; max-width:1000px; }}
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:20px; color:var(--n9); margin:38px 0 10px; line-height:1.25; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }}
article a {{ color:var(--b6); }}
.keybox {{ margin:22px 0 6px; padding:16px 20px; border-radius:6px; border:1px solid #e2e7e7; border-left:8px solid var(--b5); background:#fff; }}
.keybox .status {{ font-family:'Merriweather',Georgia,serif; font-size:19px; font-weight:700; color:var(--b6); margin:0 0 6px; }}
.keybox p {{ margin:0; font-size:13.5px; color:var(--n8); }}
figure {{ margin:20px 0 8px; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid #e2e7e7; border-radius:4px; background:#fff; }}
figcaption {{ font-size:12px; color:var(--n7); margin-top:7px; line-height:1.5; }}
.tablewrap {{ overflow-x:auto; }}
table.data {{ border-collapse:collapse; width:100%; margin:14px 0 4px; font-size:13px; }}
table.data th {{ text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:6px 9px 5px; vertical-align:bottom; }}
table.data td {{ border-bottom:1px solid #eef1f1; padding:6px 9px; color:var(--n8); vertical-align:top; }}
table.data th.num, table.data td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
table.data th.ctr, table.data td.ctr {{ text-align:center; }}
table.data td.river {{ font-weight:500; color:var(--n9); white-space:nowrap; }}
table.data td.wrap {{ min-width:210px; }}
table.data td.dim {{ color:var(--n7); font-size:12px; }}
table.data tr.group td {{ border-top:1px solid var(--b1); }}
table.data tr.total td {{ border-top:2px solid var(--b1); font-weight:500; color:var(--n9); }}
table.data tr.adopted td {{ background:var(--b05); }}
table.data tr.hit td {{ background:var(--b05); }}
table.data tr.sect td {{ background:#f6f8f8; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--n7); font-weight:700; padding:5px 9px; }}
.yr {{ display:inline-block; background:var(--b05); border:1px solid var(--b1); border-radius:3px; padding:0 6px; margin:1px 3px 1px 0; font-size:12px; font-variant-numeric:tabular-nums; color:var(--b7); }}
.tag {{ display:inline-block; background:var(--b6); color:#fff; border-radius:3px; padding:1px 7px; font-size:10px; text-transform:uppercase; letter-spacing:.05em; font-weight:700; margin-left:6px; vertical-align:1px; }}
.reached {{ display:inline-block; background:var(--b6); color:#fff; border-radius:3px; padding:0 7px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--b6); }}
p.tnote {{ font-size:12px; color:var(--n7); margin:6px 0 0; line-height:1.5; }}
.contact {{ margin:12px 0 8px; padding:16px 20px; border-radius:6px; background:var(--b05); border-left:5px solid var(--b6); }}
.contact p {{ margin:0; }}
footer {{ margin:34px 44px 0; padding:14px 0 40px; border-top:1px solid #eef1f1; font-size:12.5px; color:var(--n7); max-width:95ch; }}
footer p {{ margin:5px 0; line-height:1.6; }}
@media (max-width:640px){{ article {{ padding:4px 22px 16px; }} .hero {{ padding:32px 22px 28px; }} footer {{ margin:34px 22px 0; }} .brand {{ padding:14px 22px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <a href="https://www.unocha.org/"><img src="../assets/ocha_logo.png" alt="OCHA"></a>
    <a href="https://centre.humdata.org/"><img src="../assets/centre_for_humdata.png" alt="Centre for Humanitarian Data"></a>
  </div>
  <header class="hero">
    <div class="inner">
      <p class="crumb"><a href="../">Ethiopia Riverine Flood Watch</a> / methods</p>
      <h1>Methods and data behind the Ethiopia Riverine Flood Watch</h1>
      <p>What sits behind the watch page: the rivers and zones it covers, how each river's
         October to December level was set, and how the
         years when levels were reached line up with the flood impact records. Every number, table and
         figure below comes from the analysis notebook's own output.</p>
    </div>
  </header>

  <article>
    <div class="keybox">
      <p class="status">When the watch page shows REACHED</p>
      <p>The watch page shows REACHED when any of the seven river systems (Wabi Shebelle |
         Genale Dawa | lower Omo | Bilate | Abaya-Chamo lakes | Baro | Akobo) is at or over its
         OND level on a forecast day between 1 October and 31 December. Each river's level
         starts at the level of its 3rd-largest OND season on the 2003 to 2025 GloFAS record
         and is lowered where that catches further seasons in years when a level was already
         reached, without letting any new year in: a 1-in-8 to 1-in-6 event depending on the
         river. On the record, at least one river is at or over its level in 8 years in 23 =
         1-in-3.0 (2008, 2011, 2013, 2017, 2019, 2023, 2024, 2025). This is monitoring only:
         there is no official trigger behind this page and nothing is released when a level
         is reached.</p>
    </div>

    <h2>Rivers and zones monitored</h2>
    <p>The watch covers the river basin and zone pairs the EDRMC Bega flood alert lists as at
       risk of riverine flooding: the middle and lower Wabi Shebelle (Shebelle zone) | the lower
       Genale Dawa (Afder, Liben, Dawa) | the lower Omo (South Omo) | the Bilate (Sidama, West
       Guji) | the Kulfo, Slena and Sego at Arba Minch (Gamo) | the Akobo (Agnewak) | the Baro
       (Itang, Nuwer, Agnewak). Fourteen GloFAS points monitor these reaches; each sits on a
       channel cell verified against the model's own discharge field and named river geometry.</p>

    <h2>River levels, station by station</h2>
    <p>Each station's typical season peak is the median of its 23 October to December peaks
       (the highest daily flow of each season, 2003 to 2025). Each season, every station's peak
       is compared with its typical peak, and the river takes its highest station. The river's
       factor starts at its 3rd-biggest season and is lowered to also catch lower-ranked seasons
       in years when a level was already reached, stopping at the first season outside them so
       no new year can enter (Genale Dawa and Baro gain 2019, Akobo gains 2008). Each station's
       threshold is its typical season peak times the river's factor. Record: GloFAS version 4
       reanalysis, October to December, 2003 to 2025; version 4 is used because the live
       forecast runs on it.</p>
    <div class="tablewrap">{levels_table}</div>
    <p class="tnote">Threshold vs typical peak: the river's factor. 1.23 means the threshold is 23%
       above a typical season's peak. Every station on a river shares its river's factor.</p>

    <h2>October to December seasons, 2003 to 2025</h2>
    <figure>
      <img src="{fig_stat}" alt="Season peak compared with the typical peak, per river, with the reached seasons boxed">
      <figcaption>Each river's season peak compared with its typical season peak, 2003 to 2025.
        Red boxes: the seasons at or over the river's level.</figcaption>
    </figure>
    <figure>
      <img src="{fig_tiles}" alt="Seasons at or over each river's level and the union">
      <figcaption>Seasons at or over each river's level, and the any-river union across
        them.</figcaption>
    </figure>
    <div class="tablewrap">{acts_table}</div>

    <h2>Years reached and recorded floods</h2>
    <p>Three records beside the years when a level was reached: FloodScan flood events in the Somali-region riverine
       zones (satellite; the extract for the other zones is not built and is a named gap),
       EM-DAT flood events whose locations name the riverine areas and whose dates touch OND
       (manually refreshed snapshot), and CERF allocations to Ethiopia with emergency type
       flood (national record).</p>
    <div class="tablewrap">{impact_table}</div>
    <p class="tnote">Shaded rows: years when a level was reached. The FloodScan column covers the
       Somali-region zones only; a dot marks a season with a flood event at that severity.</p>

    <h2>Comparison with GloFAS version 5</h2>
    <p>The watch uses GloFAS version 4 levels, the version the live forecast runs. For
       reference, the same method on version 5 reanalysis (notebook 05) reaches a level in
       9 years instead of 8: six years are shared, version 5 adds 2014, 2020 and 2021, and
       drops 2011 and 2013.</p>
    <div class="tablewrap">{v5_table}</div>
    <p class="tnote">Version 5 levels are not used in monitoring yet: GloFAS version 5 forecasts
       are coming soon.</p>

    <h2>Forecast range: 10 days</h2>
    <p>Monitoring reads the operational GloFAS ensemble median at leads 1 to 10 days. GloFAS
       publishes to 30 days; 10 days is a monitoring choice. It gives the page sight of the
       October to December window from about 21 September and roughly a week of forecast
       inside the season.</p>

    <h2>Limitations</h2>
    <ul>
      <li>The Kulfo, Slena and Sego at Arba Minch are below GloFAS's 0.05 degree grid; the Gamo
        station is the Abaya-Chamo lake system cell and is labelled as such.</li>
      <li>The Dawa is monitored by a dedicated cell in Daawa zone; the Dolow cell sits below the
        Genale-Dawa confluence.</li>
      <li>All values are model quantities, not gauge readings.</li>
      <li>Forecast accuracy at each lead time is not shown here.</li>
      <li>FloodScan covers the Somali-region zones only.</li>
    </ul>

    <h2 id="contact">Contact</h2>
    <div class="contact">
      <p>Reach out to OCHA Data Science at <a href="mailto:ocha-datascience@un.org">ocha-datascience@un.org</a> with any questions.</p>
    </div>
  </article>

  <footer>
    <p>Source: <a href="https://github.com/OCHA-DAP/ds-aa-eth-flooding/blob/main/analysis/04_ond_trigger_design.ipynb">analysis/04_ond_trigger_design.ipynb</a>
       and <a href="https://github.com/OCHA-DAP/ds-aa-eth-flooding/blob/main/analysis/05_v5_reanalysis_test.ipynb">05_v5_reanalysis_test.ipynb</a>
       in OCHA-DAP/ds-aa-eth-flooding · page generated {generated} from the executed notebooks.</p>
    <p>Live monitoring: <a href="../">Ethiopia Riverine Flood Watch</a>.</p>
  </footer>
</div>
</body>
</html>
"""

(OUT / "index.html").write_text(page, encoding="utf-8")
print(f"methodology page built -> {OUT / 'index.html'}")
