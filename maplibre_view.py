# -*- coding: utf-8 -*-
"""
Painel de mapa interativo em um quadrado, no estilo do "Explore" do dynamical.org.

Dois motores, no mesmo HTML autocontido:
- **Canvas 2D próprio (padrão)**: não depende de internet, WebGL, Web Workers nem de servidores de mapa.
  Arrastar, zoom (roda/pinça/botões), terra e contorno de todos os países, nomes, grade de coordenadas,
  campo da previsão, setas de vento, legenda e valor sob o cursor/clique.
- **MapLibre + OpenFreeMap (opcional)**: mapa-base online com ruas e rótulos. Se falhar (CDN, WebGL, Worker,
  tempo esgotado), o painel volta sozinho ao motor Canvas e avisa o motivo.
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

# nome -> (motor, url do estilo, tema dos painéis)
BASEMAPS = {
    "Escuro": ("canvas", None, "dark"),
    "Claro": ("canvas", None, "light"),
    "Online escuro (OpenFreeMap)": ("style", "https://tiles.openfreemap.org/styles/dark", "dark"),
    "Online claro (OpenFreeMap)": ("style", "https://tiles.openfreemap.org/styles/positron", "light"),
    "Online ruas (OpenFreeMap)": ("style", "https://tiles.openfreemap.org/styles/liberty", "light"),
}
DEFAULT_BASEMAP = "Escuro"
ATTRIBUTION_ONLINE = ('<a href="https://openfreemap.org" target="_blank">OpenFreeMap</a> '
                      '© <a href="https://www.openmaptiles.org/" target="_blank">OpenMapTiles</a> · '
                      'Dados © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> · '
                      'Países: Natural Earth')
ATTRIBUTION_OFFLINE = "Países: Natural Earth (domínio público)"


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
    if lon.size and lon.max() > 180:  # grade 0–360
        lon = ((lon + 180.0) % 360.0) - 180.0
    uniq, idx = np.unique(lon, return_index=True)  # ordena e remove duplicatas (0° e 360°)
    if not np.array_equal(idx, np.arange(lon.size)):
        lon, field = uniq, field[:, idx]
        u, v = (u[:, idx], v[:, idx]) if u is not None else (None, None)
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


def field_overlay(prep, lut, smooth=True):
    """PNG do campo, reamostrado em linhas uniformes de Web Mercator, e seus limites (w, s, e, n)."""
    lat, lon, f = prep["lat"], prep["lon"], prep["field"]
    nlat, nlon = f.shape
    dlat = (lat[-1] - lat[0]) / (nlat - 1) if nlat > 1 else 1.0
    dlon = (lon[-1] - lon[0]) / (nlon - 1) if nlon > 1 else 1.0
    south, north = max(lat[0] - dlat / 2, -MERC_MAX), min(lat[-1] + dlat / 2, MERC_MAX)
    west, east = float(lon[0] - dlon / 2), float(lon[-1] + dlon / 2)  # bordas das células (grade global = 360° exatos)
    ys, yn = float(_merc_y(south)), float(_merc_y(north))
    nrows = int(min(max(round(nlat * (yn - ys) / math.radians(north - south)), 2), 2400))
    yy = yn - (np.arange(nrows) + 0.5) * (yn - ys) / nrows
    lat_r = np.degrees(2 * np.arctan(np.exp(yy)) - math.pi / 2)
    pos = (lat_r - lat[0]) / dlat
    idx = np.clip(np.rint(pos).astype(int), 0, nlat - 1)
    img = near = f[idx, :]
    if smooth and nlat > 1:  # interpolação linear em latitude: sem blocos nas altas latitudes (Mercator estica)
        i0 = np.clip(np.floor(pos).astype(int), 0, nlat - 1)
        i1 = np.clip(i0 + 1, 0, nlat - 1)
        t = np.clip(pos - i0, 0, 1)[:, None]
        a, b = f[i0, :], f[i1, :]
        img = np.where(np.isfinite(a) & np.isfinite(b) & np.isfinite(near), a * (1 - t) + b * t, near)
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
# Dados embutidos no HTML (formato plano, compacto)
# --------------------------------------------------------------------------
def rings_payload(bbox, margin=3.0):
    """Anéis (terra/contorno) de todos os países que tocam o bbox (S, N, W, E), como listas planas [lon, lat, ...]."""
    s, n, w, e = bbox
    span = max(n - s, e - w)
    out = []
    for c in MV._countries("50m"):
        for a, (x0, y0, x1, y1) in c["rings"]:
            if x1 < w - margin or x0 > e + margin or y1 < s - margin or y0 > n + margin:
                continue
            if span > 120 and len(a) > 150:  # mapa global: simplifica anéis longos para aliviar o HTML
                a = np.vstack([a[::2], a[-1:]])
            ring = np.column_stack([a[:, 0], np.clip(a[:, 1], -85.0, 85.0)])
            out.append(np.round(ring.ravel(), 2).tolist())
    return out


def labels_payload():
    """Nomes de países em português: [nome, lon, lat, rank], do mais ao menos importante."""
    items = [[c["name"], c["lx"], c["ly"], c["rank"]] for c in MV._countries("50m")]
    return sorted(items, key=lambda x: x[3])


def arrows_lines(lon, lat, u, v, target=24):
    """Setas como polilinhas planas [lon, lat, ...] (ou None)."""
    g = arrows_geojson(lon, lat, u, v, target)
    if g is None:
        return None
    return [np.asarray(c, float).ravel().round(3).tolist() for c in g["geometry"]["coordinates"]]


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#box{position:relative;width:min(100vw,__SIZE__px);aspect-ratio:1/1;border-radius:14px;overflow:hidden;border:1px solid rgba(127,127,127,.35)}
body.dark #box{background:#0b0f19} body.light #box{background:#eef2f8}
#map{position:absolute;inset:0}
#cv{position:absolute;inset:0;width:100%;height:100%;display:none;cursor:grab;touch-action:none}
#cv.dragging{cursor:grabbing}
.panel{position:absolute;z-index:3;padding:8px 10px;border-radius:10px;font-size:12px;line-height:1.4}
body.dark .panel{background:rgba(11,15,25,.80);color:#e8eef9;border:1px solid rgba(255,255,255,.12)}
body.light .panel{background:rgba(255,255,255,.90);color:#1b2433;border:1px solid rgba(0,0,0,.10)}
#title{top:10px;left:10px;max-width:56%}
#title b{display:block;font-size:13px}
#title span{display:block;opacity:.75}
#readout{margin-top:4px;font-variant-numeric:tabular-nums}
#zoomctl{top:10px;right:10px;padding:0;display:flex;flex-direction:column;overflow:hidden}
#zoomctl button{all:unset;cursor:pointer;width:32px;height:32px;line-height:32px;text-align:center;font-size:18px;font-weight:600}
#zoomctl button+button{border-top:1px solid rgba(127,127,127,.3)}
#zoomctl button:hover{background:rgba(127,127,127,.2)}
#legend{left:10px;bottom:30px;width:220px}
#bar{height:10px;border-radius:5px;margin:4px 0 2px}
#ticks{display:flex;justify-content:space-between;font-size:11px;opacity:.85}
#unit{opacity:.75;font-size:11px}
#ctrl{right:10px;bottom:30px;display:flex;flex-direction:column;gap:6px;min-width:150px}
#ctrl label{display:flex;align-items:center;gap:6px;cursor:pointer}
#ctrl input[type=range]{width:100%}
#attrib{right:8px;bottom:6px;padding:2px 8px;border-radius:8px;font-size:10px;opacity:.9;max-width:94%}
#attrib a{color:inherit}
#popup{display:none;z-index:4;pointer-events:auto;min-width:120px}
#popup b{font-size:13px} #popup .x{position:absolute;top:2px;right:6px;cursor:pointer;opacity:.6}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:24px;z-index:9}
body.dark #msg{background:#0b0f19;color:#e8eef9} body.light #msg{background:#eef2f8;color:#1b2433}
.maplibregl-popup-content{font:12px Inter,system-ui,sans-serif;color:#1b2433}
</style></head>
<body>
<div id="box">
  <div id="map"></div>
  <canvas id="cv"></canvas>
  <div id="title" class="panel"><b id="t1"></b><span id="t2"></span><div id="readout">Passe o mouse sobre o mapa</div></div>
  <div id="zoomctl" class="panel"><button id="zin" title="Aproximar">+</button><button id="zout" title="Afastar">−</button><button id="zhome" title="Enquadrar os dados">⌂</button></div>
  <div id="legend" class="panel"><div id="bar"></div><div id="ticks"></div><div id="unit"></div></div>
  <div id="ctrl" class="panel">
    <label>Opacidade <input id="opacity" type="range" min="0.1" max="1" step="0.05"></label>
    <label><input id="bordersChk" type="checkbox" checked> Contorno dos países</label>
    <label id="arrowsRow"><input id="arrowsChk" type="checkbox" checked> Setas de vento</label>
  </div>
  <div id="popup" class="panel"></div>
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
(function(){ const t = $("ticks"), n = 5;
  for (let i = 0; i < n; i++) { const s = document.createElement("span");
    s.textContent = String(Number((CFG.vmin + (CFG.vmax - CFG.vmin) * i / (n - 1)).toPrecision(3))); t.appendChild(s); } })();
if (!CFG.arrows) $("arrowsRow").style.display = "none";
$("attrib").innerHTML = CFG.attribution;
function note(msg){ $("t2").textContent = CFG.subtitle + " · " + msg; }
function fatal(msg){ const m = $("msg"); m.textContent = msg; m.style.display = "flex"; }

// ---------- grade de valores (leitura sob o cursor)
const grid = CFG.grid;
const data = (function(){ const bin = atob(grid.b64), n = bin.length, bytes = new Uint8Array(n);
  for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i); return new Float32Array(bytes.buffer); })();
function valueAt(lng, lat){
  while (lng < -180) lng += 360; while (lng >= 180) lng -= 360;
  const i = Math.round((lat - grid.lat0) / grid.dlat);
  let j = Math.round((lng - grid.lon0) / grid.dlon);
  if (grid.wrap) j = ((j % grid.nlon) + grid.nlon) % grid.nlon;
  if (i < 0 || i >= grid.nlat || j < 0 || j >= grid.nlon) return NaN;
  return data[i * grid.nlon + j];
}
function fmt(v){ return isNaN(v) ? "sem dado" : String(Number(v.toPrecision(4))) + (CFG.units ? " " + CFG.units : ""); }
function coordText(lng, lat){ while (lng < -180) lng += 360; while (lng >= 180) lng -= 360; return lat.toFixed(2) + "°, " + lng.toFixed(2) + "°"; }
function readout(lng, lat){ $("readout").textContent = coordText(lng, lat) + " → " + fmt(valueAt(lng, lat)); }

// ---------- projeção Web Mercator (coordenadas de mundo 0..1)
const MAXLAT = 85.0511;
function mx(lon){ return (lon + 180) / 360; }
function my(lat){ lat = Math.max(-MAXLAT, Math.min(MAXLAT, lat)); const s = Math.sin(lat * Math.PI / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); }
function invY(y){ const n = Math.PI - 2 * Math.PI * y; return 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n))); }
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const state = {opacity: CFG.opacity, borders: true, arrows: !!CFG.arrows};
let engine = null;
$("opacity").addEventListener("input", function(e){ state.opacity = parseFloat(e.target.value); if (engine) engine.setOpacity(state.opacity); });
$("bordersChk").addEventListener("change", function(e){ state.borders = e.target.checked; if (engine) engine.setBorders(state.borders); });
$("arrowsChk").addEventListener("change", function(e){ state.arrows = e.target.checked; if (engine) engine.setArrows(state.arrows); });
$("zin").addEventListener("click", function(){ if (engine) engine.zoomBy(1); });
$("zout").addEventListener("click", function(){ if (engine) engine.zoomBy(-1); });
$("zhome").addEventListener("click", function(){ if (engine) engine.home(); });

// =====================================================================
// Motor 1: Canvas 2D próprio (sem internet / WebGL / Workers)
// =====================================================================
function canvasEngine(){
  const cv = $("cv"), ctx = cv.getContext("2d");
  const TH = dark
    ? {ocean: "#0b0f19", land: "#192135", border: "rgba(255,255,255,0.62)", halo: "rgba(5,8,15,0.55)", grid: "rgba(255,255,255,0.07)",
       gridText: "rgba(255,255,255,0.38)", label: "rgba(232,238,249,0.90)", labelHalo: "rgba(11,15,25,0.85)", arrow: "rgba(255,255,255,0.92)"}
    : {ocean: "#eef2f8", land: "#dde3ee", border: "rgba(20,30,50,0.65)", halo: "rgba(255,255,255,0.55)", grid: "rgba(0,0,0,0.07)",
       gridText: "rgba(0,0,0,0.40)", label: "rgba(27,36,51,0.88)", labelHalo: "rgba(238,242,248,0.85)", arrow: "rgba(10,15,30,0.88)"};
  function toWorld(flat){
    const n = flat.length / 2, X = new Float64Array(n), Y = new Float64Array(n);
    let x0 = 9, x1 = -9, y0 = 9, y1 = -9;
    for (let i = 0; i < n; i++) { const x = mx(flat[2 * i]), y = my(flat[2 * i + 1]); X[i] = x; Y[i] = y;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return {X: X, Y: Y, n: n, x0: x0, x1: x1, y0: y0, y1: y1};
  }
  const rings = CFG.rings.map(toWorld);
  const arrows = (CFG.arrows || []).map(toWorld);
  const labels = CFG.labels.map(function(l){ return {n: l[0], x: mx(l[1]), y: my(l[2]), r: l[3]}; });
  const img = new Image(); let imgReady = false;
  img.onload = function(){ imgReady = true; schedule(); };
  img.src = CFG.png;
  const IW0 = mx(CFG.coords[0][0]), IW1 = mx(CFG.coords[1][0]), IH0 = my(CFG.coords[0][1]), IH1 = my(CFG.coords[3][1]);
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  let W = 720, H = 720, raf = 0, pendingHome = false;
  const view = {cx: 0.5, cy: 0.5, z: 1};

  function measure(){
    const r = cv.getBoundingClientRect();
    const w = Math.round(r.width) || cv.clientWidth || 0, h = Math.round(r.height) || cv.clientHeight || 0;
    return [w, h];
  }
  function resize(){
    const m = measure(); W = Math.max(1, m[0]); H = Math.max(1, m[1]);
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const worldPx = () => 256 * Math.pow(2, view.z);
  function constrain(){
    view.z = clamp(view.z, 0, 11); const Wd = worldPx();
    view.cx = ((view.cx % 1) + 1) % 1;
    view.cy = (Wd >= H) ? clamp(view.cy, H / (2 * Wd), 1 - H / (2 * Wd)) : 0.5;
  }
  function home(){
    const pad = 28, dx = Math.max(IW1 - IW0, 1e-6), dy = Math.max(IH1 - IH0, 1e-6);
    view.z = Math.min(Math.log2(Math.max(W - 2 * pad, 10) / (256 * dx)), Math.log2(Math.max(H - 2 * pad, 10) / (256 * dy)));
    view.cx = (IW0 + IW1) / 2; view.cy = (IH0 + IH1) / 2; constrain(); closePopup(); schedule();
  }
  function schedule(){ if (!raf) raf = requestAnimationFrame(function(){ raf = 0; draw(); }); }

  function graticule(Wd, xmin, xmax, ymin, ymax){
    const degPerPx = 360 / Wd; let step = 30;
    for (const s of [30, 15, 10, 5, 2, 1, 0.5, 0.25, 0.1]) { if (s / degPerPx >= 90) step = s; }
    ctx.lineWidth = 1; ctx.strokeStyle = TH.grid; ctx.fillStyle = TH.gridText; ctx.font = "10px Inter, system-ui, sans-serif";
    const lonMin = xmin * 360 - 180, lonMax = xmax * 360 - 180;
    const latTop = invY(clamp(ymin, 0, 1)), latBot = invY(clamp(ymax, 0, 1));
    ctx.beginPath();
    for (let lon = Math.ceil(lonMin / step) * step; lon <= lonMax; lon += step) {
      const x = Math.round((mx(lon) - view.cx) * Wd + W / 2) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let lat = Math.ceil(latBot / step) * step; lat <= latTop; lat += step) {
      const y = Math.round((my(lat) - view.cy) * Wd + H / 2) + 0.5; ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    for (let lat = Math.ceil(latBot / step) * step; lat <= latTop; lat += step) {
      const y = (my(lat) - view.cy) * Wd + H / 2; if (y > 40 && y < H - 40) ctx.fillText(lat.toFixed(step < 1 ? 1 : 0) + "°", 4, y + 2); }
    ctx.textBaseline = "bottom";
    for (let lon = Math.ceil(lonMin / step) * step; lon <= lonMax; lon += step) {
      let l = lon; while (l > 180) l -= 360; while (l < -180) l += 360;
      const x = (mx(lon) - view.cx) * Wd + W / 2; if (x > 40 && x < W - 40) ctx.fillText(l.toFixed(step < 1 ? 1 : 0) + "°", x + 3, H - 20); }
  }
  function tracePath(list, k, Wd, xmin, xmax, ymin, ymax, close){
    for (const r of list) {
      if (r.x1 + k < xmin || r.x0 + k > xmax || r.y1 < ymin || r.y0 > ymax) continue;
      ctx.moveTo((r.X[0] + k - view.cx) * Wd + W / 2, (r.Y[0] - view.cy) * Wd + H / 2);
      for (let i = 1; i < r.n; i++) ctx.lineTo((r.X[i] + k - view.cx) * Wd + W / 2, (r.Y[i] - view.cy) * Wd + H / 2);
      if (close) ctx.closePath();
    }
  }
  function drawLabels(Wd, xmin, xmax, ymin, ymax){
    const span = 360 * W / Wd, maxRank = span > 180 ? -1 : span > 120 ? 1 : span > 70 ? 2 : span > 35 ? 4 : span > 15 ? 6 : 10;
    if (maxRank < 0) return;
    ctx.font = "600 11px Inter, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineWidth = 3; ctx.lineJoin = "round"; const boxes = [];
    for (const l of labels) {
      if (l.r > maxRank || l.y < ymin || l.y > ymax) continue;
      for (let k = Math.floor(xmin); k <= Math.floor(xmax); k++) {
        const x = (l.x + k - view.cx) * Wd + W / 2, y = (l.y - view.cy) * Wd + H / 2;
        if (x < 30 || x > W - 30 || y < 12 || y > H - 12) continue;
        const w = ctx.measureText(l.n).width + 6, h = 14; let clash = false;
        for (const b of boxes) { if (Math.abs(b.x - x) < (b.w + w) / 2 && Math.abs(b.y - y) < (b.h + h) / 2) { clash = true; break; } }
        if (clash) continue; boxes.push({x: x, y: y, w: w, h: h});
        ctx.strokeStyle = TH.labelHalo; ctx.strokeText(l.n, x, y); ctx.fillStyle = TH.label; ctx.fillText(l.n, x, y);
      }
    }
  }
  function draw(){
    const Wd = worldPx();
    ctx.globalAlpha = 1; ctx.fillStyle = TH.ocean; ctx.fillRect(0, 0, W, H);
    const xmin = view.cx - W / (2 * Wd), xmax = view.cx + W / (2 * Wd), ymin = view.cy - H / (2 * Wd), ymax = view.cy + H / (2 * Wd);
    graticule(Wd, xmin, xmax, ymin, ymax);
    const k0 = Math.floor(xmin), k1 = Math.floor(xmax);
    ctx.fillStyle = TH.land; ctx.beginPath();
    for (let k = k0; k <= k1; k++) tracePath(rings, k, Wd, xmin, xmax, ymin, ymax, true);
    ctx.fill();
    if (imgReady) {
      ctx.globalAlpha = state.opacity; ctx.imageSmoothingEnabled = CFG.resampling !== "nearest";
      for (let k = k0; k <= k1; k++) ctx.drawImage(img, (IW0 + k - view.cx) * Wd + W / 2, (IH0 - view.cy) * Wd + H / 2,
        (IW1 - IW0) * Wd, (IH1 - IH0) * Wd);
      ctx.globalAlpha = 1;
    }
    if (state.borders) {
      ctx.lineJoin = "round"; ctx.beginPath();
      for (let k = k0; k <= k1; k++) tracePath(rings, k, Wd, xmin, xmax, ymin, ymax, true);
      const lw = view.z < 2 ? 0.55 : view.z < 4 ? 0.75 : 0.9;  // linhas mais finas em mapas amplos
      ctx.lineWidth = lw + 1.2; ctx.strokeStyle = TH.halo; ctx.stroke();
      ctx.lineWidth = lw; ctx.strokeStyle = TH.border; ctx.stroke();
    }
    if (state.arrows && arrows.length) {
      ctx.beginPath(); for (let k = k0; k <= k1; k++) tracePath(arrows, k, Wd, xmin, xmax, ymin, ymax, false);
      ctx.lineWidth = 1.1; ctx.strokeStyle = TH.arrow; ctx.stroke();
    }
    drawLabels(Wd, xmin, xmax, ymin, ymax);
  }

  // ---------- interação: arrastar, zoom (roda, pinça, duplo clique), clique
  function pos(e){ const r = cv.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
  function toLngLat(px, py){ const Wd = worldPx();
    const wy = (py - H / 2) / Wd + view.cy;
    return {lng: ((px - W / 2) / Wd + view.cx) * 360 - 180, lat: invY(wy), inside: wy >= 0 && wy <= 1}; }
  function zoomAt(px, py, dz){
    const Wd = worldPx(), wx = (px - W / 2) / Wd + view.cx, wy = (py - H / 2) / Wd + view.cy;
    view.z = clamp(view.z + dz, 0, 11); const Wd2 = worldPx();
    view.cx = wx - (px - W / 2) / Wd2; view.cy = wy - (py - H / 2) / Wd2; constrain(); closePopup(); schedule();
  }
  function closePopup(){ $("popup").style.display = "none"; }
  function showPopup(px, py){
    const ll = toLngLat(px, py), p = $("popup");
    if (!ll.inside) { closePopup(); return; }
    p.innerHTML = "<span class='x' id='px'>×</span><b></b><br><span></span>";
    p.querySelector("b").textContent = fmt(valueAt(ll.lng, ll.lat)); p.querySelector("span:last-child").textContent = coordText(ll.lng, ll.lat);
    p.style.left = clamp(px + 12, 4, W - 150) + "px"; p.style.top = clamp(py + 12, 4, H - 70) + "px"; p.style.display = "block";
    p.querySelector(".x").addEventListener("click", closePopup);
  }
  const ptrs = new Map(); let moved = 0, pinch = null;
  cv.addEventListener("pointerdown", function(e){
    const p = pos(e); ptrs.set(e.pointerId, p); closePopup(); moved = 0; cv.classList.add("dragging");
    if (cv.setPointerCapture && e.pointerId !== undefined) { try { cv.setPointerCapture(e.pointerId); } catch (x) {} }
    if (ptrs.size === 2) { const a = Array.from(ptrs.values()); pinch = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y); }
  });
  cv.addEventListener("pointermove", function(e){
    const p = pos(e), prev = ptrs.get(e.pointerId);
    if (prev) {
      if (ptrs.size === 2) {
        ptrs.set(e.pointerId, p); const a = Array.from(ptrs.values()), d = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
        if (pinch > 0 && d > 0) zoomAt((a[0].x + a[1].x) / 2, (a[0].y + a[1].y) / 2, Math.log2(d / pinch)); pinch = d; moved += 10;
      } else {
        const Wd = worldPx(); view.cx -= (p.x - prev.x) / Wd; view.cy -= (p.y - prev.y) / Wd; constrain();
        moved += Math.abs(p.x - prev.x) + Math.abs(p.y - prev.y); ptrs.set(e.pointerId, p); schedule();
      }
    }
    const ll = toLngLat(p.x, p.y);
    if (ll.inside) readout(ll.lng, ll.lat); else $("readout").textContent = "fora do mapa";
  });
  function up(e){
    const p = pos(e); const had = ptrs.delete(e.pointerId);
    if (ptrs.size === 0) { cv.classList.remove("dragging"); pinch = null; if (had && moved < 5) showPopup(p.x, p.y); }
  }
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", function(e){ ptrs.delete(e.pointerId); cv.classList.remove("dragging"); });
  cv.addEventListener("wheel", function(e){ e.preventDefault(); const p = pos(e);
    zoomAt(p.x, p.y, -e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0025)); }, {passive: false});
  cv.addEventListener("dblclick", function(e){ const p = pos(e); zoomAt(p.x, p.y, 1); });

  resize(); if (W < 50) pendingHome = true; home();
  const onResize = function(){ const m = measure(); if (m[0] !== W || m[1] !== H) { resize(); if (pendingHome && W >= 50) { pendingHome = false; home(); } else { constrain(); schedule(); } } };
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(onResize).observe($("box"));
  window.addEventListener("resize", onResize);
  return {kind: "canvas", view: view, draw: draw, home: home,
    zoomBy: function(dz){ zoomAt(W / 2, H / 2, dz); },
    setOpacity: schedule, setBorders: schedule, setArrows: schedule};
}

function startCanvas(why){
  $("map").style.display = "none"; $("cv").style.display = "block";
  if (why) note(why);
  engine = canvasEngine(); window.__engine = engine;
}

// =====================================================================
// Motor 2: MapLibre + OpenFreeMap (opcional; cai para o Canvas se falhar)
// =====================================================================
function loadAsset(kind, urls, ok, fail){
  let i = 0;
  (function next(){
    if (i >= urls.length) { fail(); return; }
    const url = urls[i++], el = document.createElement(kind === "js" ? "script" : "link");
    if (kind === "js") { el.src = url; el.async = true; } else { el.rel = "stylesheet"; el.href = url; }
    el.onload = ok; el.onerror = next; document.head.appendChild(el);
  })();
}
function toPairs(flat){ const o = []; for (let i = 0; i < flat.length; i += 2) o.push([flat[i], flat[i + 1]]); return o; }
function overlayStyle(style){  // injeta o campo, contornos e setas NO estilo: não depende do evento 'load'
  const s = JSON.parse(JSON.stringify(style));
  delete s.center; delete s.zoom; delete s.bearing; delete s.pitch;  // não deixa o estilo sobrescrever o enquadramento
  s.sources = s.sources || {}; s.layers = s.layers || [];
  const mine = [];
  s.sources.field = {type: "image", url: CFG.png, coordinates: CFG.coords};
  mine.push({id: "field", type: "raster", source: "field",
    paint: {"raster-opacity": state.opacity, "raster-resampling": CFG.resampling, "raster-fade-duration": 0}});
  s.sources.borders = {type: "geojson", data: {type: "Feature", properties: {},
    geometry: {type: "MultiLineString", coordinates: CFG.rings.map(toPairs)}}};
  mine.push({id: "borders-halo", type: "line", source: "borders",
    paint: {"line-color": dark ? "rgba(5,8,15,0.55)" : "rgba(255,255,255,0.55)", "line-width": 2.2}});
  mine.push({id: "borders", type: "line", source: "borders",
    paint: {"line-color": dark ? "rgba(255,255,255,0.70)" : "rgba(20,30,50,0.70)", "line-width": 0.8}});
  if (CFG.arrows) {
    s.sources.arrows = {type: "geojson", data: {type: "Feature", properties: {},
      geometry: {type: "MultiLineString", coordinates: CFG.arrows.map(toPairs)}}};
    mine.push({id: "arrows", type: "line", source: "arrows",
      paint: {"line-color": dark ? "rgba(255,255,255,0.92)" : "rgba(10,15,30,0.88)", "line-width": 1.1}});
  }
  let idx = s.layers.findIndex(l => l.type === "symbol"); if (idx < 0) idx = s.layers.length;
  s.layers.splice.apply(s.layers, [idx, 0].concat(mine));
  return s;
}
function startOnline(){
  return new Promise(function(resolve, reject){
    const fail = why => reject(new Error(why));
    const V = CFG.maplibreVersion;
    const BASE = ["https://unpkg.com/maplibre-gl@" + V + "/dist/", "https://cdn.jsdelivr.net/npm/maplibre-gl@" + V + "/dist/"];
    function build(style){
      let map;
      try {
        map = new maplibregl.Map({container: "map", style: overlayStyle(style), bounds: CFG.view,
          fitBoundsOptions: {padding: 24, maxZoom: 9}, dragRotate: false, pitchWithRotate: false,
          attributionControl: false, minZoom: 0, maxZoom: 12});
      } catch (e) { return fail("WebGL/MapLibre indisponível: " + (e && e.message)); }
      if (map.touchZoomRotate && map.touchZoomRotate.disableRotation) map.touchZoomRotate.disableRotation();
      let loaded = false;
      const kill = () => { try { map.remove(); } catch (e) {} };
      const timer = setTimeout(function(){ if (!loaded) { kill(); fail("o mapa online não carregou em 9 s"); } }, 9000);
      map.on("error", function(e){
        const m = String((e && e.error && e.error.message) || "");
        if (!loaded && /webgl|worker/i.test(m)) { clearTimeout(timer); kill(); fail(m); }
        else if (window.console) console.warn("maplibre:", m);
      });
      const fit = () => { try { map.resize(); map.fitBounds(CFG.view, {padding: 24, duration: 0, maxZoom: 9}); } catch (e) {} };
      map.on("load", function(){
        loaded = true; clearTimeout(timer); fit(); setTimeout(fit, 400);
        if (typeof ResizeObserver !== "undefined") new ResizeObserver(function(){ try { map.resize(); } catch (e) {} }).observe($("box"));
        const vis = (ids, on) => ids.forEach(id => { try { map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); } catch (e) {} });
        resolve({kind: "online", map: map, home: fit,
          zoomBy: dz => map.zoomTo(map.getZoom() + dz, {duration: 200}),
          setOpacity: v => { try { map.setPaintProperty("field", "raster-opacity", v); } catch (e) {} },
          setBorders: on => vis(["borders-halo", "borders"], on), setArrows: on => vis(["arrows"], on)});
      });
      map.on("mousemove", function(e){ readout(e.lngLat.lng, e.lngLat.lat); });
      map.on("click", function(e){
        new maplibregl.Popup({closeButton: true, maxWidth: "260px"}).setLngLat(e.lngLat)
          .setHTML("<b>" + fmt(valueAt(e.lngLat.lng, e.lngLat.lat)) + "</b><br>" + coordText(e.lngLat.lng, e.lngLat.lat)).addTo(map); });
      window.__map = map;
    }
    function go(){
      const ctrl = (typeof AbortController !== "undefined") ? new AbortController() : null;
      const to = setTimeout(function(){ if (ctrl) ctrl.abort(); }, 7000);
      fetch(CFG.basemap.url, ctrl ? {signal: ctrl.signal} : {})
        .then(function(r){ clearTimeout(to); if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
        .then(build, function(){ fail("estilo do mapa-base indisponível"); });
    }
    loadAsset("css", BASE.map(b => b + "maplibre-gl.css"), function(){}, function(){});
    if (window.maplibregl) go();
    else loadAsset("js", BASE.map(b => b + "maplibre-gl.js"), go, function(){ fail("não foi possível carregar o MapLibre"); });
  });
}

// ---------- partida
if (CFG.basemap.kind === "canvas") { startCanvas(); }
else {
  $("readout").textContent = "Carregando o mapa online…";
  startOnline().then(function(e){ engine = e; window.__engine = e; $("readout").textContent = "Passe o mouse sobre o mapa"; })
    .catch(function(err){ startCanvas("mapa online indisponível (" + (err && err.message) + "); usando o modo offline"); });
}
})();
</script></body></html>
"""


def build_html(data, *, title, subtitle="", colorscale=None, basemap=DEFAULT_BASEMAP, opacity=0.85,
               resampling="linear", robust=True, show_arrows=True, size=720):
    """HTML autocontido do painel quadrado; devolve (html, info)."""
    kind, url, theme = BASEMAPS.get(basemap, BASEMAPS[DEFAULT_BASEMAP])
    prep = prepare_field(data, robust=robust)
    cs = colorscale if colorscale is not None else MV.default_colorscale(data["key"])
    lut = colormap_lut(cs)
    png, (w, s, e, n) = field_overlay(prep, lut, smooth=resampling != "nearest")
    lat, lon, f = prep["lat"], prep["lon"], prep["field"]
    nlat, nlon = f.shape
    dlat = (lat[-1] - lat[0]) / (nlat - 1) if nlat > 1 else 1.0
    dlon = (lon[-1] - lon[0]) / (nlon - 1) if nlon > 1 else 1.0
    arrows = None
    if show_arrows and prep["u"] is not None:
        arrows = arrows_lines(lon, lat, prep["u"], prep["v"])
    stops = ["#%02x%02x%02x" % tuple(int(c) for c in lut[int(round(i * 255 / 8))]) for i in range(9)]
    cfg = {
        "title": title, "subtitle": subtitle, "units": data.get("units", ""), "theme": theme,
        "basemap": {"kind": kind, "url": url}, "maplibreVersion": MAPLIBRE_VERSION,
        "attribution": ATTRIBUTION_ONLINE if kind == "style" else ATTRIBUTION_OFFLINE,
        "png": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        "coords": [[w, n], [e, n], [e, s], [w, s]], "view": [[max(w, -180.0), s], [min(e, 180.0), n]],
        "vmin": prep["vmin"], "vmax": prep["vmax"], "stops": stops, "opacity": float(opacity),
        "resampling": resampling, "rings": rings_payload((s, n, w, e)), "labels": labels_payload(), "arrows": arrows,
        "grid": {"lat0": float(lat[0]), "dlat": float(dlat), "nlat": int(nlat), "lon0": float(lon[0]),
                 "dlon": float(dlon), "nlon": int(nlon), "wrap": bool(abs(lon[-1] - lon[0]) + dlon >= 359.0),
                 "b64": base64.b64encode(np.ascontiguousarray(f, dtype="<f4").tobytes()).decode("ascii")},
    }
    blob = json.dumps(cfg, separators=(",", ":"), allow_nan=False, default=float).replace("</", "<\\/")
    html = _TEMPLATE.replace("__SIZE__", str(int(size))).replace("__CONFIG__", blob)
    return html, {"vmin": prep["vmin"], "vmax": prep["vmax"], "stride": prep["stride"], "bytes": len(html)}
