"""
Build docs/design/index.html: the OND trigger design page, generated from the
executed notebook 04. Figures (PNG) and result tables (pandas HTML) are
harvested straight from analysis/04_ond_trigger_design.ipynb so the page
always shows exactly what the notebook computed. Rerun after rerunning the
notebook.
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "04_ond_trigger_design.ipynb"
OUT = HERE.parent / "docs" / "design"
FIGS = OUT / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(exist_ok=True)

nb = json.loads(NB.read_text(encoding="utf-8"))


def cell_outputs(marker: str) -> dict:
    """Outputs of the first code cell whose source contains the marker."""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return c.get("outputs", [])
    raise KeyError(marker)


def html_of(marker: str) -> str:
    for o in cell_outputs(marker):
        if o.get("output_type") == "execute_result" and "text/html" in o.get("data", {}):
            return "".join(o["data"]["text/html"])
    raise KeyError(f"no html table in cell: {marker}")


def png_of(marker: str, name: str) -> str:
    for o in cell_outputs(marker):
        if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
            (FIGS / name).write_bytes(base64.b64decode(o["data"]["image/png"]))
            return f"figs/{name}"
    raise KeyError(f"no figure in cell: {marker}")


def stream_of(marker: str) -> str:
    txt = ""
    for o in cell_outputs(marker):
        if o.get("output_type") == "stream":
            txt += "".join(o["text"])
    return txt.strip()


levels_html = html_of('levels["threshold_m3s"]')
fig_stat = png_of("Season statistic per river", "season_statistic.png")
fig_tiles = png_of("Activation seasons per river", "activations.png")
acts_text = stream_of("per-river activation seasons")
options_html = html_of("pd.DataFrame(opts)")
lead_text = stream_of("POD by lead")
impact_html = html_of('impact["cerf_usd"]')

generated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethiopia OND Trigger Design</title>
<meta name="description" content="Design and backtest of the OND riverine flood trigger for Ethiopia: seven river systems, per-river top-3 levels, overall activation frequency 1-in-3.">
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
article h2 {{ font-family:'Merriweather',Georgia,serif; font-size:20px; color:var(--n9); margin:34px 0 10px; line-height:1.25; }}
article p, article li {{ font-size:14.5px; color:var(--n8); line-height:1.65; }}
article a {{ color:var(--b6); }}
.keybox {{ margin:22px 0 6px; padding:16px 20px; border-radius:6px; border:1px solid #e2e7e7; border-left:8px solid var(--b5); background:#fff; }}
.keybox .status {{ font-family:'Merriweather',Georgia,serif; font-size:19px; font-weight:700; color:var(--b6); margin:0 0 6px; }}
.keybox p {{ margin:0; font-size:13.5px; color:var(--n8); }}
figure {{ margin:20px 0 8px; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid #e2e7e7; border-radius:4px; background:#fff; }}
figcaption {{ font-size:12px; color:var(--n7); margin-top:7px; line-height:1.5; }}
.tablewrap {{ overflow-x:auto; }}
table.dataframe {{ border-collapse:collapse; width:100%; margin:12px 0 4px; font-size:12.5px; border:none; }}
table.dataframe th {{ text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:6px 8px 5px; border-top:none; border-left:none; border-right:none; }}
table.dataframe td {{ border:none; border-bottom:1px solid #eef1f1; padding:5px 8px; color:var(--n8); white-space:nowrap; font-variant-numeric:tabular-nums; }}
pre.mono {{ background:#f6f8f8; border:1px solid #e2e7e7; border-radius:4px; padding:12px 14px; font-size:12px; line-height:1.5; overflow-x:auto; color:var(--n8); }}
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
      <p class="crumb"><a href="../">Ethiopia Riverine Flood Watch</a> / trigger design</p>
      <h1>Ethiopia OND Trigger Design</h1>
      <p>How the OND riverine flood trigger was designed and backtested: the areas it covers,
         how the levels were set, the alternatives considered, and the trigger's record against
         the flood impact records. Generated from the analysis notebook; every table and figure
         below is the notebook's own output.</p>
    </div>
  </header>

  <article>
    <div class="keybox">
      <p class="status">The trigger</p>
      <p>Reached when any of the seven river systems (Wabi Shebelle | Genale Dawa | lower Omo |
         Bilate | Abaya-Chamo lakes | Baro | Akobo) is at or over its OND trigger level on a
         forecast day between 1 October and 31 December. Each river's level is the level of its
         3rd-largest OND season on the 2003 to 2025 GloFAS record: a 1-in-8 event per river.
         Overall activation frequency: 8 years in 23 = 1-in-3.0 (2008, 2011, 2013, 2017, 2019,
         2023, 2024, 2025).</p>
    </div>

    <h2>Where</h2>
    <p>The trigger covers the river basin and zone pairs the EDRMC Bega flood alert lists as at
       risk of riverine flooding: the middle and lower Wabi Shebelle (Shebelle zone) | the lower
       Genale Dawa (Afder, Liben, Dawa) | the lower Omo (South Omo) | the Bilate (Sidama, West
       Guji) | the Kulfo, Slena and Sego at Arba Minch (Gamo) | the Akobo (Agnewak) | the Baro
       (Itang, Nuwer, Agnewak). Fourteen GloFAS points monitor these reaches; each sits on a
       channel cell verified against the model's own discharge field and named river geometry.</p>

    <h2>Levels</h2>
    <p>A river's season statistic is the maximum over its stations of the station's OND daily
       peak divided by its median OND seasonal maximum, which lets stations of very different
       size carry equal weight inside one system. The river's threshold ratio is its 3rd-largest
       season statistic; each station's level in m³/s is that ratio times the station's median
       OND seasonal maximum. Record: GloFAS v4 reanalysis at channel-snapped cells, October to
       December, 2003 to 2025. v4 is pinned because the operational forecast runs at v4 scale.</p>
    <div class="tablewrap">{levels_html}</div>

    <h2>The rivers' flood seasons</h2>
    <figure>
      <img src="{fig_stat}" alt="Season statistic per river with the top 3 seasons boxed">
      <figcaption>Each river's normalised seasonal peak, 2003 to 2025. Red boxes: the river's
        top 3 seasons, which define its level.</figcaption>
    </figure>
    <figure>
      <img src="{fig_tiles}" alt="Activation seasons per river and the union">
      <figcaption>Activation seasons per river and the any-river union.</figcaption>
    </figure>
    <pre class="mono">{acts_text}</pre>

    <h2>How the depth was chosen</h2>
    <p>The working group's constraints were per-river ranking, an any-river trigger, and an
       overall activation frequency of 1-in-3. At a per-river top third the union activates in
       18 of 23 years (1-in-1.3). The candidates compared:</p>
    <div class="tablewrap">{options_html}</div>
    <p>The adopted rule is the first row: the only candidate meeting the overall 1-in-3 exactly
       while keeping the any-river shape and per-river ranking. Its per-river bar is 1-in-8
       rather than the top third; the working group accepted that trade on 29 September 2026.</p>

    <h2>Impact record, year by year</h2>
    <p>Three records beside the trigger: FloodScan flood events in the Somali-region riverine
       zones (satellite; the extract for the other zones is not built and is a named gap),
       EM-DAT flood events whose locations name the riverine areas and whose dates touch OND
       (manually refreshed snapshot), and CERF allocations to Ethiopia with emergency type
       flood (national record).</p>
    <div class="tablewrap">{impact_html}</div>

    <h2>Lead time</h2>
    <p>Monitoring reads the operational GloFAS ensemble median at leads 1 to 10 days. GloFAS
       publishes to 30 days; 10 days is a monitoring choice, not an optimised value. It gives
       the page sight of the OND window from about 21 September and roughly a week of warning
       inside the season. The archive evidence (v4 reforecast 2003 to 2023, Somali stations,
       scored against FloodScan events) shows forecast skill flat across leads 1 to 7: these
       rivers carry initial-condition memory rather than rainfall-forecast skill.</p>
    <pre class="mono">{lead_text}</pre>
    <p>Two gaps, stated: leads 8 to 10 are not validated (the reforecast archive was fetched at
       leads 1 to 7 only), and no lead is validated for the Omo, Bilate, Abaya-Chamo lakes,
       Baro or Akobo (no reforecast was fetched for those basins).</p>

    <h2>Coverage notes</h2>
    <ul>
      <li>The Kulfo, Slena and Sego at Arba Minch are below GloFAS's 0.05 degree grid; the Gamo
        station is the Abaya-Chamo lake system cell and is labelled as such.</li>
      <li>The Dawa is monitored by a dedicated cell in Daawa zone; the Dolow cell sits below the
        Genale-Dawa confluence.</li>
      <li>All values are model quantities, not gauge readings. FloodScan validation for the
        non-Somali zones is an open gap.</li>
    </ul>

    <h2 id="contact">Contact</h2>
    <div class="contact">
      <p>Reach out to OCHA Data Science at <a href="mailto:ocha-datascience@un.org">ocha-datascience@un.org</a> with any questions.</p>
    </div>
  </article>

  <footer>
    <p>Source: <a href="https://github.com/OCHA-DAP/ds-aa-eth-flooding/blob/main/analysis/04_ond_trigger_design.ipynb">analysis/04_ond_trigger_design.ipynb</a>
       in OCHA-DAP/ds-aa-eth-flooding · page generated {generated} from the executed notebook.</p>
    <p>Live monitoring: <a href="../">Ethiopia Riverine Flood Watch</a>.</p>
  </footer>
</div>
</body>
</html>
"""

(OUT / "index.html").write_text(page, encoding="utf-8")
print(f"design page built -> {OUT / 'index.html'}")
