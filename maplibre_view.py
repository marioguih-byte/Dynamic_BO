# -*- coding: utf-8 -*-
"""
Painel de mapa interativo (MapLibre GL JS) em um quadrado, no estilo do "Explore" do dynamical.org.

- Mapa-base real com zoom/arrastar (OpenFreeMap, sem chave) ou modo "simples" offline (terra + países embutidos)
- Campo da previsão desenhado por cima (PNG reprojetado para Web Mercator), com legenda
- Valor sob o cursor / clique, contornos de todos os países, setas de vento, controle de opacidade
- HTML autocontido (usa só CDN do MapLibre e o estilo do OpenFreeMap no navegador)
"""
from __future__ import annotations

import base64
import json
import math
import re
import struct
import zlib

import numpy as np
import plotly.colors as pc

import mapview as MV

MERC_MAX = 85.0511
MAPLIBRE_VERSION = "4.7.1"

# nome -> (tipo, url do estilo, tema dos painéis)
BASEMAPS = {
    "Escuro (OpenFreeMap)": ("style", "https://tiles.openfreemap.org/styles/dark", "dark"),
    "Claro (OpenFreeMap)": ("style", "https://tiles.openfreemap.org/styles/positron", "light"),
    "Ruas (OpenFreeMap Liberty)": ("style", "https://tiles.openfreemap.org/styles/liberty", "light"),
    "Simples escuro (offline)": ("simple", None, "dark"),
    "Simples claro (offline)": ("simple", None, "light"),
}
ATTRIBUTION_ONLINE = ('<a href="https://openfreemap.org" target="_blank">OpenFreeMap</a> '
                      '© <a href="https://www.openmaptiles.org/" target="_blank">OpenMapTiles</a> · '
                      'Dados © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> · '
                      'Países: Natural Earth')
ATTRIBUTION_SIMPLE = 'Países: Natural Earth (domínio público)'


# --------------------------------------------------------------------------
# Cores e PNG
# --------------------------------------------------------------------------
def _parse_color(c):
    c = c.strip()
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(x * 2 for x in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgba?\(([^)]+)\)", c)
    parts = [float(x) for x in m.group(1).split(",")][:3]
    return tuple(int(round(p)) for p in parts)


def colormap_lut(colorscale, n=256):
    """Tabela (n, 3) uint8 a partir de um nome do Plotly (ex.: 'Turbo', 'RdYlBu_r') ou lista [[pos, cor], ...]."""
    cs = pc.get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
    pos = np.array([float(p) for p, _ in cs])
    cols = np.array([_parse_color(c) for _, c in cs], float)
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)


def png_rgba(rgba):
    """PNG RGBA 8 bits sem dependências (zlib + struct)."""
    h, w, _ = rgba.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgba.reshape(h, w * 4)], axis=1).tobytes()

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _merc_y(lat):
    lat = np.clip(lat, -MERC_MAX, MERC_MAX)
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


# --------------------------------------------------------------------------
# Preparação do campo
# --------------------------------------------------------------------------
def prepare_field(data, robust=True, max_cells=900_000):
    """Grade regular ascendente (lat, lon), campo mascarado e limites da escala de cores."""
    key, units = data["key"], data.get("units", "")
    lat, lon = np.asarray(data["lat"], float), np.asarray(data["lon"], float)
    field = np.asarray(data["field"], float)
    u = np.asarray(data["u"], float) if "u" in data else None
    v = np.asarray(data["v"], float) if "v" in data else None
    if lat.ndim == 2:  # grade curvilínea -> regular (vizinho mais próximo)
        try:
            lon, lat, outs = MV._regrid(lat, lon, [field] + ([u, v] if u is not None else []))
        except ImportError as exc:
            raise RuntimeError("Grades curvilíneas precisam do pacote scipy (pip install scipy).") from exc
        field = outs[0]
        u, v = (outs[1], outs[2]) if u is not None else (None, None)
    if lat[0] > lat[-1]:
        lat, field = lat[::-1], field[::-1]
        u, v = (u[::-1], v[::-1]) if u is not None else (None, None)
    if lon[0] > lon[-1]:
        lon, field = lon[::-1], field[:, ::-1]
        u, v = (u[:, ::-1], v[:, ::-1]) if u is not None else (None, None)
    k = max(1, math.ceil(math.sqrt(field.size / max_cells)))
    if k > 1:  # limita o peso do HTML
        lat, lon, field = lat[::k], lon[::k], field[::k, ::k]
        u, v = (u[::k, ::k], v[::k, ::k]) if u is not None else (None, None)
    if MV._is_precip_key(key) and units in ("mm", "mm/h"):
        field = np.where(field <= 0.05, np.nan, field)
    fin = field[np.isfinite(field)]
    vmin, vmax = 0.0, 1.0
    if fin.size:
        vmin, vmax = (float(x) for x in (np.percentile(fin, [1, 99]) if robust else (fin.min(), fin.max())))
        if not vmax > vmin:
            vmin, vmax = float(fin.min()), float(fin.max()) + 1e-9
    return dict(lat=lat, lon=lon, field=field, u=u, v=v, vmin=vmin, vmax=vmax, stride=k)


def field_overlay(prep, lut):
    """PNG do campo, reamostrado em linhas uniformes de Web Mercator, e seus limites (w, s, e, n)."""
    lat, lon, f = prep["lat"], prep["lon"], prep["field"]
    nlat, nlon = f.shape
    dlat = (lat[-1] - lat[0]) / (nlat - 1) if nlat > 1 else 1.0
    dlon = (lon[-1] - lon[0]) / (nlon - 1) if nlon > 1 else 1.0
    south, north = max(lat[0] - dlat / 2, -MERC_MAX), min(lat[-1] + dlat / 2, MERC_MAX)
    west, east = max(lon[0] - dlon / 2, -180.0), min(lon[-1] + dlon / 2, 180.0)
    ys, yn = float(_merc_y(south)), float(_merc_y(north))
    nrows = int(min(max(round(nlat * (yn - ys) / math.radians(north - south)), 2), 2400))
    yy = yn - (np.arange(nrows) + 0.5) * (yn - ys) / nrows
    lat_r = np.degrees(2 * np.arctan(np.exp(yy)) - math.pi / 2)
    idx = np.clip(np.rint((lat_r - lat[0]) / dlat).astype(int), 0, nlat - 1)
    img = f[idx, :]
    t = np.clip((img - prep["vmin"]) / (prep["vmax"] - prep["vmin"]), 0, 1)
    rgb = lut[np.nan_to_num(t * 255).round().astype(np.uint8)]
    alpha = np.where(np.isfinite(img), 255, 0).astype(np.uint8)
    return png_rgba(np.dstack([rgb, alpha])), (float(west), float(south), float(east), float(north))


# --------------------------------------------------------------------------
# Países e setas (GeoJSON)
# --------------------------------------------------------------------------
def borders_land_geojson(bbox, margin=3.0):
    """(linhas, polígonos) dos países que tocam o bbox (S, N, W, E)."""
    s, n, w, e = bbox
    res = MV._resolution(bbox)
    lines, polys = [], []
    for c in MV._countries(res):
        for a, (x0, y0, x1, y1) in c["rings"]:
            if x1 < w - margin or x0 > e + margin or y1 < s - margin or y0 > n + margin:
                continue
            ring = np.column_stack([a[:, 0], np.clip(a[:, 1], -85.0, 85.0)]).tolist()
            lines.append(ring)
            polys.append([ring])
    feat = lambda g: {"type": "Feature", "properties": {}, "geometry": g}  # noqa: E731
    return (feat({"type": "MultiLineString", "coordinates": lines}),
            feat({"type": "MultiPolygon", "coordinates": polys}))


def arrows_geojson(lon, lat, u, v, target=24):
    """Setas de vento em lon/lat, com forma correta em Web Mercator (ou None se não houver vento)."""
    ny, nx = u.shape
    ky, kx = max(1, round(ny / target)), max(1, round(nx / target))
    xs, ys = lon[::kx], lat[::ky]
    uu, vv = u[::ky, ::kx], v[::ky, ::kx]
    X, Y = np.meshgrid(xs, ys)
    sp = np.hypot(uu, vv)
    ok = np.isfinite(sp) & (sp > 0)
    if not ok.any():
        return None
    cosphi = np.cos(np.radians(np.clip(Y, -80, 80)))
    dlon = abs(float(np.median(np.diff(xs)))) if len(xs) > 1 else 1.0
    dlat = abs(float(np.median(np.diff(ys)))) if len(ys) > 1 else 1.0
    lmax = 0.8 * np.minimum(dlon, dlat / cosphi)
    ref = float(np.nanpercentile(sp[ok], 95)) or 1.0
    L = lmax * np.clip(sp / ref, 0.3, 1.0)
    th = np.arctan2(vv, uu)
    X0, Y0, L, th, c = X[ok], Y[ok], L[ok], th[ok], cosphi[ok]
    sx, sy = L * np.cos(th), L * np.sin(th)  # espaço de tela (graus de longitude)
    tail = np.stack([X0 - sx / 2, Y0 - sy / 2 * c], -1)
    tip = np.stack([X0 + sx / 2, Y0 + sy / 2 * c], -1)
    head = 0.34 * L
    a1, a2 = th + math.radians(155), th - math.radians(155)
    h1 = np.stack([tip[:, 0] + head * np.cos(a1), tip[:, 1] + head * np.sin(a1) * c], -1)
    h2 = np.stack([tip[:, 0] + head * np.cos(a2), tip[:, 1] + head * np.sin(a2) * c], -1)
    seg1, seg2 = np.stack([tail, tip, h1], 1), np.stack([tip, h2], 1)
    seg1[..., 1], seg2[..., 1] = np.clip(seg1[..., 1], -85, 85), np.clip(seg2[..., 1], -85, 85)
    coords = seg1.round(3).tolist() + seg2.round(3).tolist()
    return {"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": coords}}


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#box{position:relative;width:min(100vw,__SIZE__px);aspect-ratio:1/1;border-radius:14px;overflow:hidden;border:1px solid rgba(127,127,127,.35)}
#map{position:absolute;inset:0}
.panel{position:absolute;z-index:3;padding:8px 10px;border-radius:10px;font-size:12px;line-height:1.4}
body.dark .panel{background:rgba(11,15,25,.80);color:#e8eef9;border:1px solid rgba(255,255,255,.12)}
body.light .panel{background:rgba(255,255,255,.90);color:#1b2433;border:1px solid rgba(0,0,0,.10)}
#title{top:10px;left:10px;max-width:56%}
#title b{display:block;font-size:13px}
#title span{display:block;opacity:.75}
#readout{margin-top:4px;font-variant-numeric:tabular-nums}
#legend{left:10px;bottom:30px;width:220px}
#bar{height:10px;border-radius:5px;margin:4px 0 2px}
#ticks{display:flex;justify-content:space-between;font-size:11px;opacity:.85}
#unit{opacity:.75;font-size:11px}
#ctrl{right:10px;bottom:30px;display:flex;flex-direction:column;gap:6px;min-width:150px}
#attrib{right:8px;bottom:6px;padding:2px 8px;border-radius:8px;font-size:10px;opacity:.9;max-width:94%}
#attrib a{color:inherit}
#ctrl label{display:flex;align-items:center;gap:6px;cursor:pointer}
#ctrl input[type=range]{width:100%}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:24px;z-index:9}
body.dark #msg{background:#0b0f19;color:#e8eef9} body.light #msg{background:#eef2f8;color:#1b2433}
.maplibregl-popup-content{font:12px Inter,system-ui,sans-serif;color:#1b2433}
</style></head>
<body>
<div id="box">
  <div id="map"></div>
  <div id="title" class="panel"><b id="t1"></b><span id="t2"></span><div id="readout">Passe o mouse sobre o mapa</div></div>
  <div id="legend" class="panel"><div id="bar"></div><div id="ticks"></div><div id="unit"></div></div>
  <div id="ctrl" class="panel">
    <label>Opacidade <input id="opacity" type="range" min="0.1" max="1" step="0.05"></label>
    <label><input id="bordersChk" type="checkbox" checked> Contorno dos países</label>
    <label id="arrowsRow"><input id="arrowsChk" type="checkbox" checked> Setas de vento</label>
  </div>
  <div id="attrib" class="panel"></div>
  <div id="msg"></div>
</div>
<script>
(function(){
const CFG = __CONFIG__;
const dark = CFG.theme === "dark";
const $ = id => document.getElementById(id);
document.body.classList.add(dark ? "dark" : "light");
$("t1").textContent = CFG.title; $("t2").textContent = CFG.subtitle;
$("unit").textContent = CFG.units ? "Unidade: " + CFG.units : "";
$("opacity").value = CFG.opacity;
$("bar").style.background = "linear-gradient(to right," + CFG.stops.join(",") + ")";
(function(){
  const t = $("ticks"), n = 5;
  for (let i = 0; i < n; i++) { const s = document.createElement("span");
    s.textContent = String(Number((CFG.vmin + (CFG.vmax - CFG.vmin) * i / (n - 1)).toPrecision(3))); t.appendChild(s); }
})();
if (!CFG.arrows) $("arrowsRow").style.display = "none";
$("attrib").innerHTML = CFG.attribution;

function fatal(msg){ const m = $("msg"); m.textContent = msg; m.style.display = "flex"; }
function note(msg){ $("t2").textContent = CFG.subtitle + " · " + msg; }

function loadAsset(kind, urls, ok, fail){
  let i = 0;
  (function next(){
    if (i >= urls.length) { fail(); return; }
    const url = urls[i++];
    const el = document.createElement(kind === "js" ? "script" : "link");
    if (kind === "js") { el.src = url; el.async = true; } else { el.rel = "stylesheet"; el.href = url; }
    el.onload = ok; el.onerror = next; document.head.appendChild(el);
  })();
}
const V = CFG.maplibreVersion;
const BASE = ["https://unpkg.com/maplibre-gl@" + V + "/dist/", "https://cdn.jsdelivr.net/npm/maplibre-gl@" + V + "/dist/"];
loadAsset("css", BASE.map(b => b + "maplibre-gl.css"), function(){}, function(){});

const grid = CFG.grid;
const data = (function(){ const bin = atob(grid.b64), n = bin.length, bytes = new Uint8Array(n);
  for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i); return new Float32Array(bytes.buffer); })();
function valueAt(ll){
  let lon = ll.lng; while (lon < -180) lon += 360; while (lon >= 180) lon -= 360;
  const i = Math.round((ll.lat - grid.lat0) / grid.dlat);
  let j = Math.round((lon - grid.lon0) / grid.dlon);
  if (grid.wrap) j = ((j % grid.nlon) + grid.nlon) % grid.nlon;
  if (i < 0 || i >= grid.nlat || j < 0 || j >= grid.nlon) return NaN;
  return data[i * grid.nlon + j];
}
function fmt(v){ return isNaN(v) ? "sem dado" : String(Number(v.toPrecision(4))) + (CFG.units ? " " + CFG.units : ""); }
function coordText(ll){ return ll.lat.toFixed(2) + "°, " + ll.lng.toFixed(2) + "°"; }

function simpleStyle(){
  return {version: 8, sources: {}, layers: [{id: "bg", type: "background",
    paint: {"background-color": dark ? "#0b0f19" : "#eef2f8"}}]};
}
function begin(style, simple){
  const map = new maplibregl.Map({container: "map", style: style, bounds: CFG.view,
    fitBoundsOptions: {padding: 16}, dragRotate: false, pitchWithRotate: false, attributionControl: false,
    minZoom: 0, maxZoom: 12});
  if (map.touchZoomRotate && map.touchZoomRotate.disableRotation) map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
  map.on("error", function(e){ if (window.console) console.warn("maplibre:", e && e.error && e.error.message); });
  let fitted = false;
  const fit = function(){ try { map.resize(); map.fitBounds(CFG.view, {padding: 24, duration: 0, maxZoom: 9}); } catch (e) {} };
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(function(){ try { map.resize(); } catch (e) {} if (!fitted) fit(); }).observe($("box")); }
  map.on("load", function(){
    fit(); fitted = true; setTimeout(fit, 400);  // o estilo pode trazer center/zoom próprios: enquadra nos dados
    const first = ((map.getStyle().layers || []).find(l => l.type === "symbol") || {}).id;
    if (simple) {
      map.addSource("land", {type: "geojson", data: CFG.land});
      map.addLayer({id: "land", type: "fill", source: "land",
        paint: {"fill-color": dark ? "#192135" : "#dde3ee"}});
    }
    map.addSource("field", {type: "image", url: CFG.png, coordinates: CFG.coords});
    map.addLayer({id: "field", type: "raster", source: "field",
      paint: {"raster-opacity": CFG.opacity, "raster-resampling": CFG.resampling, "raster-fade-duration": 0}}, first);
    map.addSource("borders", {type: "geojson", data: CFG.borders});
    map.addLayer({id: "borders-halo", type: "line", source: "borders",
      paint: {"line-color": dark ? "rgba(5,8,15,0.55)" : "rgba(255,255,255,0.55)", "line-width": 2.2}}, first);
    map.addLayer({id: "borders", type: "line", source: "borders",
      paint: {"line-color": dark ? "rgba(255,255,255,0.70)" : "rgba(20,30,50,0.70)", "line-width": 0.8}}, first);
    if (CFG.arrows) {
      map.addSource("arrows", {type: "geojson", data: CFG.arrows});
      map.addLayer({id: "arrows", type: "line", source: "arrows",
        paint: {"line-color": dark ? "rgba(255,255,255,0.92)" : "rgba(10,15,30,0.88)", "line-width": 1.1}}, first);
    }
    $("opacity").addEventListener("input", function(e){
      map.setPaintProperty("field", "raster-opacity", parseFloat(e.target.value)); });
    $("bordersChk").addEventListener("change", function(e){
      ["borders-halo", "borders"].forEach(id => map.setLayoutProperty(id, "visibility", e.target.checked ? "visible" : "none")); });
    $("arrowsChk").addEventListener("change", function(e){
      if (CFG.arrows) map.setLayoutProperty("arrows", "visibility", e.target.checked ? "visible" : "none"); });
  });
  map.on("mousemove", function(e){ $("readout").textContent = coordText(e.lngLat) + " → " + fmt(valueAt(e.lngLat)); });
  map.on("click", function(e){
    new maplibregl.Popup({closeButton: true, maxWidth: "260px"}).setLngLat(e.lngLat)
      .setHTML("<b>" + fmt(valueAt(e.lngLat)) + "</b><br>" + coordText(e.lngLat)).addTo(map); });
  window.__map = map;
}
function start(){
  if (CFG.basemap.kind === "style") {
    const ctrl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    const to = setTimeout(function(){ if (ctrl) ctrl.abort(); }, 7000);
    fetch(CFG.basemap.url, ctrl ? {signal: ctrl.signal} : {})
      .then(function(r){ clearTimeout(to); if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function(style){ begin(style, false); })
      .catch(function(){ note("mapa-base online indisponível; usando o modo simples"); begin(simpleStyle(), true); });
  } else { begin(simpleStyle(), true); }
}
if (window.maplibregl) { start(); }
else { loadAsset("js", BASE.map(b => b + "maplibre-gl.js"), start,
  function(){ fatal("Não foi possível carregar o MapLibre (CDN bloqueado?). Use a visão Plotly."); }); }
})();
</script></body></html>
"""


def build_html(data, *, title, subtitle="", colorscale=None, basemap="Escuro (OpenFreeMap)", opacity=0.85,
               resampling="linear", robust=True, show_arrows=True, size=720):
    """HTML autocontido do painel quadrado; devolve (html, info)."""
    kind, url, theme = BASEMAPS.get(basemap, BASEMAPS["Escuro (OpenFreeMap)"])
    prep = prepare_field(data, robust=robust)
    cs = colorscale if colorscale is not None else MV.default_colorscale(data["key"])
    lut = colormap_lut(cs)
    png, (w, s, e, n) = field_overlay(prep, lut)
    lat, lon, f = prep["lat"], prep["lon"], prep["field"]
    nlat, nlon = f.shape
    dlat = (lat[-1] - lat[0]) / (nlat - 1) if nlat > 1 else 1.0
    dlon = (lon[-1] - lon[0]) / (nlon - 1) if nlon > 1 else 1.0
    bbox = (s, n, w, e)
    borders, land = borders_land_geojson(bbox)
    arrows = None
    if show_arrows and prep["u"] is not None:
        arrows = arrows_geojson(lon, lat, prep["u"], prep["v"])
    stops = ["#%02x%02x%02x" % tuple(int(c) for c in lut[int(round(i * 255 / 8))]) for i in range(9)]
    cfg = {
        "title": title, "subtitle": subtitle, "units": data.get("units", ""), "theme": theme,
        "basemap": {"kind": kind, "url": url}, "maplibreVersion": MAPLIBRE_VERSION,
        "attribution": ATTRIBUTION_ONLINE if kind == "style" else ATTRIBUTION_SIMPLE,
        "png": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        "coords": [[w, n], [e, n], [e, s], [w, s]], "view": [[w, s], [e, n]],
        "vmin": prep["vmin"], "vmax": prep["vmax"], "stops": stops, "opacity": float(opacity),
        "resampling": resampling, "borders": borders, "land": land, "arrows": arrows,
        "grid": {"lat0": float(lat[0]), "dlat": float(dlat), "nlat": int(nlat), "lon0": float(lon[0]),
                 "dlon": float(dlon), "nlon": int(nlon), "wrap": bool(abs(lon[-1] - lon[0]) + dlon >= 359.0),
                 "b64": base64.b64encode(np.ascontiguousarray(f, dtype="<f4").tobytes()).decode("ascii")},
    }
    blob = json.dumps(cfg, separators=(",", ":"), allow_nan=False, default=float).replace("</", "<\\/")
    html = _TEMPLATE.replace("__SIZE__", str(int(size))).replace("__CONFIG__", blob)
    return html, {"vmin": prep["vmin"], "vmax": prep["vmax"], "stride": prep["stride"], "bytes": len(html)}
