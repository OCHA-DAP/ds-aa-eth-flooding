"""
Build docs/rainfall/index.html: October to December rainfall over the Wabi
Shebele, Genale Dawa and Omo Gibe basins, flooding in their river zones
(1 October to 31 January) and GloFAS river flow, 1998-2025.

Reads the outputs of basin_rainfall_ond.py and the basin outlines from blob;
writes one self-contained page (data embedded, charts drawn in the browser).
"""

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd
from scipy.stats import spearmanr

STAGE = "dev"
P = "ds-aa-eth-flooding/"
OUT = Path(__file__).resolve().parent.parent / "docs" / "rainfall"
OUT.mkdir(parents=True, exist_ok=True)
BASINS = ["Wabi Shebele", "Genale Dawa", "Omo Gibe"]

years = stratus.load_csv_from_blob(P + "processed/rainfall/basin_ond_years.csv", stage=STAGE)
weights = stratus.load_csv_from_blob(P + "processed/rainfall/basin_zone_weights.csv", stage=STAGE)
pt_thr = stratus.load_csv_from_blob(P + "processed/rainfall/possible_trigger_thresholds.csv", stage=STAGE)
pt_ev = stratus.load_csv_from_blob(P + "processed/rainfall/possible_trigger_events.csv", stage=STAGE)
pt_yr = stratus.load_csv_from_blob(P + "processed/rainfall/possible_trigger_years.csv", stage=STAGE)
zc = stratus.load_csv_from_blob(P + "processed/rainfall/zone_flood_correlation.csv", stage=STAGE)
ZONE_PCODES = ["ET0506", "ET0508", "ET0509", "ET0511", "ET0707"]
basins = gpd.read_file(stratus.load_blob_data(P + "processed/basins/eth_rainfall_basins.geojson",
                                              stage=STAGE, container_name="projects")).to_crs(4326)
raw = stratus.load_blob_data("eth_shp.zip", stage=STAGE, container_name="polygon")
tmp = tempfile.mkdtemp()
zipfile.ZipFile(io.BytesIO(raw.getvalue() if hasattr(raw, "getvalue") else raw)).extractall(tmp)
adm0 = gpd.read_file(os.path.join(tmp, "eth_adm0.shp")).to_crs(4326)
adm2 = gpd.read_file(os.path.join(tmp, "eth_adm2.shp")).to_crs(4326)
used = adm2[adm2.ADM2_PCODE.isin(weights.ADM2_PCODE)]


def rings(geom, tol):
    g = geom.simplify(tol, preserve_topology=True)
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    return [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys if p.area > tol * tol * 4]


g03 = years[years.year >= 2003]
glofas_corr = [{"basin": b,
                "rain7": round(float(spearmanr(g.glofas_ratio, g.max_7day_mm).statistic), 2),
                "total": round(float(spearmanr(g.glofas_ratio, g.ond_total_mm).statistic), 2),
                "flood": round(float(spearmanr(g.glofas_ratio, g.flood_peak_pct).statistic), 2)}
               for b in BASINS for g in [g03[g03.basin == b]]]

payload = {
    "basins": [{"name": r.basin, "area_km2": int(r.area_km2), "source": r.source, "rings": rings(r.geometry, 0.02)}
               for r in basins.itertuples()],
    "country": rings(adm0.union_all(), 0.03),
    "zones": [{"name": r.ADM2_EN, "rings": rings(r.geometry, 0.02), "river": r.ADM2_PCODE in ZONE_PCODES} for r in used.itertuples()],
    "zone_corr": zc.to_dict("records"),
    "zones_per_basin": weights.groupby("basin").ADM2_PCODE.nunique().to_dict(),
    "years": {b: years[years.basin == b].sort_values("year").drop(columns="basin").astype(object)
              .where(lambda d: d.notna(), None).to_dict("records") for b in BASINS},
    "glofas_corr": glofas_corr,
    "trigger": {"thresholds": pt_thr.to_dict("records"),
                "events": pt_ev.astype(object).where(pt_ev.notna(), None).to_dict("records"),
                "years": pt_yr.to_dict("records")},
}

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Basin Rainfall and Flooding</title>
<meta name="description" content="October to December rainfall over the Wabi Shebele, Genale Dawa and Omo Gibe basins in Ethiopia, flooding in the zones along these rivers and GloFAS river flow, 1998 to 2025.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root { --b5:#1f69b3; --b6:#1f69b3; --b7:#144372; --b05:#ebf0f4; --b1:#d4e5f7;
        --n9:#1f2324; --n8:#3f4748; --n7:#5e6a6b; --n05:#f5f7f7; }
* { box-sizing:border-box; }
body { margin:0; background:var(--n05); color:var(--n9); font-family:'Roboto',system-ui,-apple-system,'Segoe UI',Arial,sans-serif; line-height:1.5; }
.wrap { max-width:1080px; margin:0 auto; background:#fff; min-height:100vh; box-shadow:0 0 40px rgba(31,35,36,.06); }
.brand { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:14px 44px; border-bottom:1px solid #e2e7e7; }
.brand a { display:block; line-height:0; } .brand img { height:44px; width:auto; display:block; }
.hero { background:var(--b6); padding:40px 44px 34px; }
.crumb { font-size:12px; margin:0 0 14px; color:rgba(255,255,255,.85); }
.crumb a { color:rgba(255,255,255,.85); text-decoration:none; } .crumb a:hover { text-decoration:underline; }
.hero h1 { font-family:'Merriweather',Georgia,serif; font-size:28px; line-height:1.2; color:#fff; margin:0 0 14px; }
.hero p { margin:0; font-size:15px; line-height:1.6; color:#fff; max-width:66ch; }
article { padding:8px 44px 48px; max-width:1000px; }
article h2 { font-family:'Merriweather',Georgia,serif; font-size:20px; margin:36px 0 10px; line-height:1.25; }
article p, article li { font-size:14.5px; color:var(--n8); line-height:1.65; }
article li { margin-bottom:6px; } article a { color:var(--b6); }
.keybox { margin:22px 0 6px; padding:16px 20px; border-radius:6px; border:1px solid #e2e7e7; border-left:8px solid var(--b5); }
.keybox .status { font-family:'Merriweather',Georgia,serif; font-size:18px; color:var(--b6); margin:0 0 8px; }
.keybox ul { margin:0; padding-left:18px; } .keybox li { font-size:14px; margin-bottom:4px; }
.maprow { display:grid; grid-template-columns:minmax(0,1fr) 250px; gap:18px; align-items:start; margin:14px 0 4px; }
@media (max-width:760px){ .maprow { grid-template-columns:1fr } }
.map { border:1px solid #e2e7e7; border-radius:6px; background:#f6f8f8; overflow:hidden; } .map svg { display:block; width:100%; height:auto; }
.legend { font-size:13px; color:var(--n8); } .legend .row { display:flex; gap:9px; align-items:flex-start; margin:0 0 12px; }
.legend .sw { width:14px; height:14px; border-radius:3px; flex:none; margin-top:3px; }
.legend .sub { font-size:12px; color:var(--n7); }
.chart { border:1px solid #e2e7e7; border-radius:5px; padding:10px 12px 6px; margin:14px 0; background:#fff; }
.chart h3 { font-family:'Merriweather',Georgia,serif; font-size:15px; margin:0 0 4px; display:flex; justify-content:space-between; gap:10px; }
.chart h3 span { font-family:'Roboto',sans-serif; font-weight:400; font-size:12px; color:var(--n7); }
.chart svg { display:block; width:100%; height:auto; }
figcaption, .note { font-size:12px; color:var(--n7); line-height:1.5; }
table.data { border-collapse:collapse; width:100%; margin:12px 0 4px; font-size:13px; }
table.data th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--n7); font-weight:700; border-bottom:2px solid var(--b1); padding:7px 9px 6px; vertical-align:bottom; }
table.data td { border-bottom:1px solid #eef1f1; padding:6px 9px; color:var(--n8); white-space:nowrap; }
table.data td.n, table.data th.n { text-align:right; font-variant-numeric:tabular-nums; }
table.data tr.group td { border-top:1px solid var(--b1); }
table.data td.basin { font-weight:500; color:var(--n9); }
.tablewrap { overflow-x:auto; }
table.data.opts td { vertical-align:top; white-space:normal; min-width:96px; }
table.data.opts td.opt { font-weight:500; color:var(--n9); min-width:140px; }
table.data.opts .sub { color:var(--n7); font-size:12px; margin-left:4px; }
table.data.opts .yrs { display:block; font-size:12px; color:var(--n8); }
table.data.opts .hits { display:block; font-size:11.5px; color:#B34036; }
.contact { margin:30px 0 8px; padding:16px 20px; border-radius:6px; background:var(--b05); border-left:5px solid var(--b6); } .contact p { margin:0; }
@media (max-width:640px){ article { padding:4px 22px 16px; } .hero { padding:32px 22px 28px; } .brand { padding:12px 22px; } .brand img { height:34px; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <a href="https://www.unocha.org/"><img src="../assets/ocha_logo.png" alt="OCHA"></a>
    <a href="https://centre.humdata.org/"><img src="../assets/centre_for_humdata.png" alt="Centre for Humanitarian Data"></a>
  </div>
  <header class="hero">
    <p class="crumb"><a href="../">Ethiopia Riverine Flood Watch</a> / rainfall and flooding</p>
    <h1>Rainfall and flooding in three river basins, 1998 to 2025</h1>
    <p>This page looks at rainfall and flooding in the Wabi Shebele, Genale Dawa and Omo Gibe basins for every
       season from 1998 to 2025. Rainfall comes from IMERG, flooding from FloodScan and river flow from GloFAS.</p>
  </header>

  <article>
    <div class="keybox"><p class="status">Summary</p><ul id="summary"></ul></div>

    <h2>Basins and zones</h2>
    <div class="maprow">
      <div class="map"><svg id="map" role="img" aria-label="Map of Ethiopia with the three river basins, the zones used for rainfall and the river zones used for flooding"></svg></div>
      <div class="legend" id="legend"></div>
    </div>
    <p class="note">Rainfall is measured over the whole basin (coloured). Flooding is measured in the zones along
       the rivers (hatched): Shabelle for Wabi Shebele, Afder, Liban and Daawa for Genale Dawa, and South Omo for
       Omo Gibe.</p>

    <h2>Rainfall and flooding by season</h2>
    <p>For each season, the top bar is the wettest 7 days over the basin and the bottom bar is the largest share
       of the river zones that was flooded on any one day. The five highest seasons are in a darker colour, and
       years in bold are in the top five for both.</p>
    <div id="charts"></div>

    <h2>Top seasons in each record, 2003 to 2025</h2>
    <div class="chart" id="strip"></div>
    <p class="note" id="stripnote"></p>

    <h2>Possible thresholds</h2>
    <p>Two possible thresholds come out of the records above: one for river flow in GloFAS and one for rainfall
       over the basin. The table shows how often each was reached in any of the three basins between 2003 and 2025. This is for monitoring
       only. It is not an official trigger and nothing is released.</p>
    <div class="tablewrap" id="trigstages"></div>
    <div class="chart" id="trigstrip"></div>
    <p class="note" id="trignote"></p>
    <div class="tablewrap" id="trigevents"></div>
    <p class="note">The days are counted to the flood peak in the river zones, from the day GloFAS reached the
       threshold or from the day the rainfall data came in, which is about two days after the rain.</p>
    <ul class="caveats">
      <li>Both thresholds were set using the same 23 seasons they are checked against.</li>
      <li id="trigsens"></li>
      <li>The river flow check uses GloFAS river flow rebuilt from past weather, not the forecasts issued at the time.</li>
    </ul>

    <h2>Data and method</h2>
    <ul>
      <li>Rainfall is IMERG Late Run daily data from NASA, taken from our database as averages for each zone, for
        1 October to 31 December. Each zone counts towards the basin in proportion to how much of it lies inside the
        basin. A 7-day total is one day's rainfall plus the six days before it.</li>
      <li>Flooding is FloodScan's standard flood extent from AER: the share of the river zones flooded each day, from
        1 October to 31 January, with each zone weighted by its area.</li>
      <li>River flow is the GloFAS version 4 reanalysis (river flow rebuilt from past weather) at the watch page's
        stations, shown as a share of each river's threshold, for 2003 to 2025.</li>
      <li>ERA5 monthly rainfall from Copernicus is used only to check 2000 (see below).</li>
      <li>Basin outlines: <span id="srcs"></span></li>
    </ul>

    <h2>Limitations</h2>
    <ul>
      <li>In IMERG, 2000 has the highest 7-day rainfall of any season in Wabi Shebele and Genale Dawa. ERA5 does not
        agree: it ranks that season 11th and 16th of 28, and FloodScan ranks the flooding 8th and 17th.</li>
      <li>In Genale Dawa, 2021 is the fifth most flooded season but the driest of all 28 for rainfall.</li>
      <li>IMERG is missing 14 to 16 November 2024 and 27 September 2025, and FloodScan is missing 25 November 2024.</li>
      <li>Zones that lie partly outside a basin bring in some rain from outside it. IMERG and FloodScan are both
        satellite estimates.</li>
    </ul>

    <h2 id="contact">Contact</h2>
    <div class="contact"><p>If you have questions about this page, contact OCHA Data Science at <a href="mailto:ocha-datascience@un.org">ocha-datascience@un.org</a>.</p></div>
  </article>
</div>
<script>
const D = __DATA__;
const COLOR = { "Wabi Shebele": "#2a78d6", "Genale Dawa": "#1baf7a", "Omo Gibe": "#eb6834" };
const BASINS = Object.keys(COLOR);
const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const day = iso => { const [y,m,d] = iso.split("-").map(Number); return d + " " + MON[m-1] + " " + y; };
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

/* summary */
const topN = (b, key, n) => D.years[b].slice().sort((a, c) => a[key] - c[key]).slice(0, n);
const both = b => new Set(topN(b, "rank_7day", 5).map(r => r.year).filter(y => topN(b, "rank_flood", 5).some(r => r.year === y)));
document.getElementById("summary").innerHTML = BASINS.map(b => {
  const list = ys => ys.slice(0, -1).join(", ") + " and " + ys[ys.length - 1];
  return `<li><b>${b}:</b> the wettest seasons were ${list(topN(b, "rank_7day", 3).map(t => t.year))}, and the most flooded ${list(topN(b, "rank_flood", 3).map(t => t.year))}. GloFAS reached its threshold in ${list(D.years[b].filter(d => d.glofas_reached).map(d => d.year))}.</li>`;
}).join("") + (() => {
  const zc = Object.fromEntries(D.zone_corr.map(c => [c.zone, c.rho_wettest7])), gv = D.glofas_corr.flatMap(c => [c.rain7, c.total, c.flood]);
  const rng = zs => { const v = zs.map(z => zc[z]); return `${Math.min(...v).toFixed(2)} to ${Math.max(...v).toFixed(2)}`; };
  return `<li>Rainfall and flooding tend to go together in Shabelle, Afder and South Omo (correlation ${rng(["Shabelle", "Afder", "South Omo"])}), much less so in Liban and Daawa (${rng(["Liban", "Daawa"])}). GloFAS river flow shows little link with either (${Math.min(...gv).toFixed(2)} to ${Math.max(...gv).toFixed(2)}).</li>`;
})();

/* map */
{
  const pts = D.country.flat();
  const lons = pts.map(p => p[0]), lats = pts.map(p => p[1]);
  const L0 = Math.min(...lons) - .2, L1 = Math.max(...lons) + .2, A0 = Math.min(...lats) - .2, A1 = Math.max(...lats) + .2;
  const W = 760, H = Math.round(W * (A1 - A0) / (L1 - L0));
  const X = x => (x - L0) / (L1 - L0) * W, Y = y => H - (y - A0) / (A1 - A0) * H;
  const path = rs => rs.map(r => r.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("") + "Z").join("");
  let g = '<path d="' + path(D.country) + '" fill="#fff" stroke="#9aa4a6" stroke-width="1"></path>';
  for (const bsn of D.basins) g += '<path d="' + path(bsn.rings) + '" fill="' + COLOR[bsn.name] + '" fill-opacity=".28" stroke="' + COLOR[bsn.name] + '" stroke-width="1.6"><title>' + esc(bsn.name) + '</title></path>';
  let g0 = '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#1f2324" stroke-width="1.1" stroke-opacity=".55"/></pattern></defs>';
  for (const z of D.zones) g += '<path d="' + path(z.rings) + '" fill="' + (z.river ? "url(#hatch)" : "none") + '" stroke="#7c8789" stroke-width=".5" stroke-opacity=".7"><title>' + esc(z.name) + '</title></path>';
  g = g0 + g;
  for (const bsn of D.basins) {
    const r = bsn.rings.reduce((a, c) => c.length > a.length ? c : a);
    const cx = r.reduce((a, p) => a + X(p[0]), 0) / r.length, cy = r.reduce((a, p) => a + Y(p[1]), 0) / r.length;
    g += '<text x="' + cx.toFixed(1) + '" y="' + cy.toFixed(1) + '" text-anchor="middle" font-size="13" font-weight="700" fill="' + COLOR[bsn.name] + '" style="paint-order:stroke;stroke:#fff;stroke-width:3px">' + esc(bsn.name) + '</text>';
  }
  const svg = document.getElementById("map"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.innerHTML = g;
  document.getElementById("legend").innerHTML = D.basins.map(bsn =>
    `<div class="row"><span class="sw" style="background:${COLOR[bsn.name]};opacity:.75"></span><span><b>${bsn.name}</b><br><span class="sub">${bsn.area_km2.toLocaleString("en-GB")} km² · ${D.zones_per_basin[bsn.name]} zones</span></span></div>`).join("");
  document.getElementById("srcs").innerHTML = D.basins.map(bsn => `${bsn.name}: ${esc(bsn.source)}`).join("; ") + ".";
}

/* charts: rainfall up, flooding down, years between */
{
  const W = 900, l = 52, r = 10, rt = 22, rh = 130, yb = 30, fh = 100, fb = 14, H = rt + rh + yb + fh + fb, iw = W - l - r;
  const rTop = Math.ceil(Math.max(...BASINS.flatMap(b => D.years[b].map(x => x.max_7day_mm))) / 20) * 20;

  const base = rt + rh, f0 = base + yb;
  const yr = v => base - v / rTop * rh;
  document.getElementById("charts").innerHTML = BASINS.map(b => {
    const rs = D.years[b], bw = iw / rs.length, both5 = both(b);
    const fTop = Math.ceil(Math.max(...rs.map(x => x.flood_peak_pct))), fstep = fTop > 4 ? 2 : fTop > 2 ? 1 : .5, yf = v => f0 + v / fTop * fh;
    let g = "";
    for (let v = 0; v <= rTop; v += 40) g += `<line x1="${l}" x2="${W - r}" y1="${yr(v).toFixed(1)}" y2="${yr(v).toFixed(1)}" stroke="#e6eaea"></line><text x="${l - 6}" y="${(yr(v) + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="#5e6a6b">${v}</text>`;
    for (let v = 0; v <= fTop + 1e-9; v += fstep) g += `<line x1="${l}" x2="${W - r}" y1="${yf(v).toFixed(1)}" y2="${yf(v).toFixed(1)}" stroke="#e6eaea"></line><text x="${l - 6}" y="${(yf(v) + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="#5e6a6b">${v}%</text>`;
    rs.forEach((d, i) => {
      const x = l + i * bw + bw * .15, w = bw * .7, cx = (x + w / 2).toFixed(1);
      g += `<rect x="${x.toFixed(1)}" y="${yr(d.max_7day_mm).toFixed(1)}" width="${w.toFixed(1)}" height="${(base - yr(d.max_7day_mm)).toFixed(1)}" fill="${d.rank_7day <= 5 ? "#1f69b3" : "#bcd0e6"}" rx="2"><title>${d.year}: ${d.max_7day_mm} mm in the 7 days to ${day(d.date_of_max)} (rank ${d.rank_7day} of 28)</title></rect>`;
      g += `<rect x="${x.toFixed(1)}" y="${f0}" width="${w.toFixed(1)}" height="${(yf(d.flood_peak_pct) - f0).toFixed(1)}" fill="${d.rank_flood <= 5 ? "#B34036" : "#eec4bf"}" rx="2"><title>${d.year}: ${d.flood_peak_pct.toFixed(2)}% of the river zones flooded on ${day(d.date_flood_peak)} (rank ${d.rank_flood} of 28)</title></rect>`;
      const k = both5.has(d.year);
      g += `<text x="${cx}" y="${base + yb / 2 + 3.5}" text-anchor="middle" font-size="${k ? 10.5 : 9.5}" font-weight="${k ? 700 : 400}" fill="${k ? "#1f2324" : "#7c8789"}">${String(d.year).slice(2)}</text>`;
    });
    g += `<text x="12" y="${(rt + rh / 2).toFixed(1)}" font-size="10" fill="#1f69b3" text-anchor="middle" transform="rotate(-90 12 ${(rt + rh / 2).toFixed(1)})">rain, mm in 7 days</text>`;
    g += `<text x="12" y="${(f0 + fh / 2).toFixed(1)}" font-size="10" fill="#B34036" text-anchor="middle" transform="rotate(-90 12 ${(f0 + fh / 2).toFixed(1)})">river zones flooded, %</text>`;
    const wet = topN(b, "rank_7day", 1)[0], fl = topN(b, "rank_flood", 1)[0];
    return `<div class="chart"><h3><span style="font-family:Merriweather,Georgia,serif;font-weight:700;font-size:15px;color:${COLOR[b]}">${b}</span><span>wettest: ${wet.year}, ${Math.round(wet.max_7day_mm)} mm | most flooded: ${fl.year}, ${fl.flood_peak_pct.toFixed(1)}%</span></h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${b}: wettest 7-day rainfall and largest flooded area each season">${g}</svg></div>`;
  }).join("");
}

/* three records, 2003-2025 */
{
  const ROWS = [["Wettest 7 days (top 5)", "#1f69b3"], ["River zone flooding (top 5)", "#B34036"], ["GloFAS threshold reached", "#1f2324"]];
  const W = 900, l = 200, r = 10, rowH = 22, top = 8, n = 23, cw = (W - l - r) / n;
  let g = "", y0 = 0;
  for (const b of BASINS) {
    const rs = D.years[b].filter(d => d.year >= 2003);
    const rk = key => new Set(rs.slice().sort((a, c) => c[key] - a[key]).slice(0, 5).map(d => d.year));
    const sets = [rk("max_7day_mm"), rk("flood_peak_pct"), new Set(rs.filter(d => d.glofas_reached).map(d => d.year))];
    g += `<text x="0" y="${y0 + 14}" font-family="Merriweather,Georgia,serif" font-weight="700" font-size="14" fill="${COLOR[b]}">${b}</text>`;
    const yb = y0 + 22;
    rs.forEach((d, i) => {
      if (sets.filter(st => st.has(d.year)).length >= 2) g += `<rect x="${(l + i * cw + 1).toFixed(1)}" y="${yb + top - 6}" width="${(cw - 2).toFixed(1)}" height="${ROWS.length * rowH + 2}" fill="#f1e9c9" rx="3"></rect>`;
    });
    ROWS.forEach(([lab, col], j) => {
      const cy = yb + top + j * rowH + 6;
      g += `<text x="${l - 8}" y="${cy + 4}" text-anchor="end" font-size="11" fill="#3f4748">${lab}</text>`;
      rs.forEach((d, i) => {
        const cx = (l + i * cw + cw / 2).toFixed(1);
        g += sets[j].has(d.year) ? `<circle cx="${cx}" cy="${cy}" r="6" fill="${col}"><title>${b} ${d.year}: ${lab.toLowerCase()}</title></circle>` : `<circle cx="${cx}" cy="${cy}" r="2" fill="#c9d0d1"></circle>`;
      });
    });
    const ly = yb + top + ROWS.length * rowH + 8;
    rs.forEach((d, i) => g += `<text x="${(l + i * cw + cw / 2).toFixed(1)}" y="${ly}" text-anchor="middle" font-size="9.5" fill="#5e6a6b">${String(d.year).slice(2)}</text>`);
    y0 = ly + 18;
  }
  document.getElementById("strip").innerHTML = `<h3>Top seasons in each record<span>shaded where two or more records agree</span></h3><svg viewBox="0 0 ${W} ${y0}" role="img" aria-label="Top seasons for rainfall, flooding and GloFAS river flow per basin, 2003 to 2025">${g}</svg>`;
  const gv = D.glofas_corr.flatMap(c => [c.rain7, c.total, c.flood]);
  document.getElementById("stripnote").textContent = `The large dots show the five wettest seasons, the five most flooded seasons, and the seasons in which GloFAS reached the river's threshold. Correlation goes from -1 to 1, and with only 23 seasons anything between about -0.4 and 0.4 could be chance.`;
}

/* possible trigger */
{
  const T = D.trigger, N = T.years.length, every = n => ((N + 1) / n).toFixed(1);
  const yrs = k => T.years.filter(d => d[k]).map(d => d.year);
  const both = T.years.filter(d => d.readiness || d.action).map(d => d.year);
  const mmv = T.thresholds.map(t => `${Math.round(t.threshold_mm)} mm in ${t.basin}`);
  const mm = mmv.slice(0, -1).join(", ") + " or " + mmv[mmv.length - 1];
  document.getElementById("trigstages").innerHTML = `<table class="data opts"><thead><tr><th>Possible threshold</th><th>What it means</th><th>Seasons</th><th>How often</th></tr></thead><tbody>
    <tr><td class="opt">River flow</td><td>GloFAS river flow reaches the river's threshold. On the watch page this is checked with the GloFAS forecast for the next 10 days.</td><td>${yrs("readiness").length}<span class="yrs">${yrs("readiness").join(", ")}</span></td><td>once every ${every(yrs("readiness").length)} years</td></tr>
    <tr><td class="opt">Rainfall</td><td>7-day rainfall over the basin reaches ${mm}. Each is the basin's second-highest season between 2003 and 2025.</td><td>${yrs("action").length}<span class="yrs">${yrs("action").join(", ")}</span></td><td>once every ${every(yrs("action").length)} years</td></tr>
    <tr><td class="opt">Either threshold</td><td></td><td>${both.length}<span class="yrs">${both.join(", ")}</span></td><td>once every ${every(both.length)} years</td></tr>
  </tbody></table>`;

  const ROWS = [["River flow threshold (GloFAS)", "readiness", "#1f2324"], ["Rainfall threshold (IMERG)", "action", "#1f69b3"],
                ["EM-DAT flood event", "emdat", "#B34036"], ["CERF allocation", "cerf", "#B34036"],
                ["FloodScan 1-in-3 flood, Somali region", "floodscan_rp3_somali", "#B34036"]];
  const W = 900, l = 230, r = 10, rowH = 22, top = 14, cw = (W - l - r) / N;
  let g = "";
  T.years.forEach((d, i) => {
    if ((d.readiness || d.action) && (d.emdat || d.cerf || d.floodscan_rp3_somali))
      g += `<rect x="${(l + i * cw + 1).toFixed(1)}" y="${top - 8}" width="${(cw - 2).toFixed(1)}" height="${ROWS.length * rowH + 2}" fill="#f1e9c9" rx="3"></rect>`;
  });
  ROWS.forEach(([lab, key, col], j) => {
    const cy = top + j * rowH + 4;
    if (j === 2) g += `<line x1="${l - 220}" x2="${W - r}" y1="${cy - rowH / 2 - 1}" y2="${cy - rowH / 2 - 1}" stroke="#d5dbdc"></line>`;
    g += `<text x="${l - 8}" y="${cy + 4}" text-anchor="end" font-size="11" fill="#3f4748">${lab}</text>`;
    T.years.forEach((d, i) => {
      const cx = (l + i * cw + cw / 2).toFixed(1);
      g += d[key] ? `<circle cx="${cx}" cy="${cy}" r="6" fill="${col}"><title>${d.year}: ${lab}</title></circle>` : `<circle cx="${cx}" cy="${cy}" r="2" fill="#c9d0d1"></circle>`;
    });
  });
  const ly = top + ROWS.length * rowH + 8;
  T.years.forEach((d, i) => g += `<text x="${(l + i * cw + cw / 2).toFixed(1)}" y="${ly}" text-anchor="middle" font-size="9.5" fill="#5e6a6b">${String(d.year).slice(2)}</text>`);
  document.getElementById("trigstrip").innerHTML = `<h3>Seasons a possible threshold was reached, compared with flood records<span>shaded where a threshold was reached and a flood was recorded</span></h3><svg viewBox="0 0 ${W} ${ly + 8}" role="img" aria-label="Seasons the river flow or rainfall threshold was reached, with EM-DAT, CERF and FloodScan flood records, 2003 to 2025">${g}</svg>`;

  const rec = d => d.emdat || d.cerf || d.floodscan_rp3_somali;
  const ra = T.years.filter(d => d.action), rr = T.years.filter(d => d.readiness);
  const part = xs => { const k = xs.filter(rec).length; return k === xs.length ? `All ${k}` : `${k} of the ${xs.length}`; };
  document.getElementById("trignote").textContent = `${part(ra)} seasons that reached the rainfall threshold and ${part(rr)} that reached the river flow threshold show up in at least one flood record. The flood records are the same as on the methods page: EM-DAT flood events in the watch's riverine areas from October to December, CERF flood allocations to Ethiopia, and FloodScan flood events in the Somali region.`;

  const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const dd = iso => { if (!iso) return ""; const [, m, d] = iso.split("-").map(Number); return d + " " + MON[m - 1]; };
  const rel = n => `${Math.abs(n)} day${Math.abs(n) === 1 ? "" : "s"} ${n >= 0 ? "before" : "after"}`;
  const when = (iso, n) => iso ? `${dd(iso)} <span class="sub">${rel(n)}</span>` : "";
  const rows = T.events.slice().sort((a, b) => a.year - b.year || BASINS.indexOf(a.basin) - BASINS.indexOf(b.basin)).map(e =>
    `<tr><td>${e.year}</td><td class="basin" style="color:${COLOR[e.basin]}">${e.basin}</td>
     <td>${e.readiness ? when(e.glofas_reached, e.days_glofas_to_flood_peak) : ""}</td>
     <td>${e.action ? when(e.rain_available, e.days_rain_available_to_flood_peak) : ""}</td>
     <td>${dd(e.flood_peak)}</td></tr>`).join("");
  document.getElementById("trigevents").innerHTML = `<table class="data opts"><thead><tr><th>Season</th><th>Basin</th><th>GloFAS reached threshold</th><th>Rainfall data came in</th><th>Flood peak in river zones</th></tr></thead><tbody>${rows}</tbody></table>`;

  // only the basins whose threshold is within 5% of the next season down
  const close = T.thresholds.filter(t => t.next_below_mm >= 0.95 * t.threshold_mm);
  const join = xs => xs.length > 1 ? xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1] : xs[0];
  document.getElementById("trigsens").textContent = close.length
    ? `The rainfall thresholds for ${join(close.map(t => t.basin))} sit close to the next season down (${join(close.map(t => `${Math.round(t.threshold_mm)} mm against ${Math.round(t.next_below_mm)} mm`))}), so a small change in the data could change which seasons count.`
    : "Each rainfall threshold is well above the next season down.";
}

</script>
</body>
</html>
"""

(OUT / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(payload)), encoding="utf-8")
print(f"rainfall page built -> {OUT / 'index.html'}")
