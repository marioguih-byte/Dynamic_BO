# -*- coding: utf-8 -*-
"""
Mapa moderno (Plotly) com contorno de TODOS os países, sem depender de cartopy.

Os limites vêm do Natural Earth (domínio público), embutidos em data/*.json.gz:
  - 110m (leve) para mapas muito amplos
  - 50m (detalhado) para regiões
"""
from __future__ import annotations

import gzip
import json
import math
import pathlib
import re
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

DATA_DIR = pathlib.Path(__file__).parent / "data"

THEMES = {
    "Escuro": dict(paper="#0b0f19", plot="#0b0f19", land="#192135", border="rgba(255,255,255,0.60)",
                   grid="rgba(255,255,255,0.07)", halo="rgba(5,8,15,0.55)", text="#e8eef9", muted="#8d99ae",
                   label="rgba(232,238,249,0.78)", arrow="rgba(255,255,255,0.92)", point="#ff4d6d"),
    "Claro": dict(paper="#ffffff", plot="#eef2f8", land="#dde3ee", border="rgba(20,30,50,0.60)",
                  grid="rgba(0,0,0,0.07)", halo="rgba(255,255,255,0.55)", text="#1b2433", muted="#667085",
                  label="rgba(27,36,51,0.72)", arrow="rgba(10,15,30,0.88)", point="#d6336c"),
}

_PRECIP = [[0.0, "#d6f0ff"], [0.2, "#7cc7ff"], [0.4, "#2f8fff"], [0.6, "#4b4ad6"], [0.8, "#9b3fc7"], [1.0, "#e0357a"]]
PALETTES = {
    "Automática": None,
    "Turbo": "Turbo", "Viridis": "Viridis", "Plasma": "Plasma", "Inferno": "Inferno", "Cividis": "Cividis",
    "Temperatura (vermelho↔azul)": "RdYlBu_r", "Chuva (azul→magenta)": _PRECIP, "Azuis": "Blues",
    "Espectral": "Spectral_r", "Cinza": "Greys_r",
}

PLOT_CONFIG = {
    "scrollZoom": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "mapa_previsao"},
}


def _is_precip_key(key):
    k = key.lower()
    return any(t in k for t in ("precip", "rain", "snowfall", "runoff")) or bool(re.search(r"(?:^|_)tp(?:_|$)", k))


def default_colorscale(key):
    k = key.lower()
    if "temp" in k or "dew" in k:
        return "RdYlBu_r"
    if _is_precip_key(k) or "snow" in k or "water" in k:
        return _PRECIP
    if "humidity" in k:
        return "YlGnBu"
    if "cloud" in k:
        return "Greys_r"
    if "wind" in k:
        return "Turbo"
    return "Viridis"


# --------------------------------------------------------------------------
# Países
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _countries(res):
    path = DATA_DIR / f"countries_{res}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)["countries"]
    out = []
    for c in raw:
        rings = []
        for flat in c["r"]:
            a = np.asarray(flat, dtype="float64").reshape(-1, 2)
            rings.append((a, (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())))
        out.append({"name": c["n"], "iso": c["a"], "lx": c["lx"], "ly": c["ly"], "rank": c["lr"], "rings": rings})
    return out


def _resolution(bbox):
    s, n, w, e = bbox
    return "110m" if max(n - s, e - w) > 70 else "50m"


def _polyline(bbox, res, margin=3.0):
    """Anéis (fechados) que tocam o bbox, concatenados com NaN entre eles."""
    s, n, w, e = bbox
    xs, ys = [], []
    for c in _countries(res):
        for a, (x0, y0, x1, y1) in c["rings"]:
            if x1 < w - margin or x0 > e + margin or y1 < s - margin or y0 > n + margin:
                continue
            xs += [a[:, 0], [np.nan]]
            ys += [a[:, 1], [np.nan]]
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _land_path(bbox, res, margin=3.0, max_pts=20000):
    """Caminho SVG (um único shape) com a terra; decimado para ficar leve."""
    s, n, w, e = bbox
    rings = []
    for c in _countries(res):
        for a, (x0, y0, x1, y1) in c["rings"]:
            if not (x1 < w - margin or x0 > e + margin or y1 < s - margin or y0 > n + margin):
                rings.append(a)
    if not rings:
        return ""
    k = max(1, math.ceil(sum(len(a) for a in rings) / max_pts))
    parts = []
    for a in rings:
        b = a[::k] if len(a) > 6 * k else a
        parts.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in b) + "Z")
    return "".join(parts)


def _labels(bbox, res):
    s, n, w, e = bbox
    span = max(n - s, e - w)
    maxrank = 10 if span <= 15 else 6 if span <= 35 else 4 if span <= 70 else 2 if span <= 120 else -1
    if maxrank < 0:
        return [], [], []
    px, py = (e - w) * 0.03, (n - s) * 0.03
    xs, ys, names = [], [], []
    for c in _countries(res):
        if c["rank"] <= maxrank and w + px <= c["lx"] <= e - px and s + py <= c["ly"] <= n - py:
            xs.append(c["lx"]); ys.append(c["ly"]); names.append(c["name"])
    return xs, ys, names


# --------------------------------------------------------------------------
# Grades curvilíneas -> grade regular (vizinho mais próximo)
# --------------------------------------------------------------------------
def _regrid(lat2, lon2, arrays, max_pts=900):
    from scipy.spatial import cKDTree
    lat0 = float(np.nanmean(lat2))
    c = math.cos(math.radians(lat0))
    pts = np.column_stack([lon2.ravel() * c, lat2.ravel()])
    ok = np.isfinite(pts).all(axis=1)
    d = float(np.nanmedian(np.hypot(np.diff(lon2, axis=1) * c, np.diff(lat2, axis=1))))
    d = d if d > 0 else 0.05
    x0, x1, y0, y1 = np.nanmin(lon2), np.nanmax(lon2), np.nanmin(lat2), np.nanmax(lat2)
    nx = int(min(max((x1 - x0) / (d / c), 2), max_pts))
    ny = int(min(max((y1 - y0) / d, 2), max_pts))
    tx, ty = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    TX, TY = np.meshgrid(tx, ty)
    tree = cKDTree(pts[ok])
    dist, idx = tree.query(np.column_stack([TX.ravel() * c, TY.ravel()]), distance_upper_bound=1.6 * d)
    miss = ~np.isfinite(dist)
    src = np.flatnonzero(ok)
    outs = []
    for a in arrays:
        flat = a.ravel()[src]
        v = np.where(miss, np.nan, flat[np.minimum(idx, len(flat) - 1)])
        outs.append(v.reshape(ny, nx))
    return tx, ty, outs


# --------------------------------------------------------------------------
# Setas de vento
# --------------------------------------------------------------------------
def _arrows(lon, lat, u, v, lat0, target=24):
    ny, nx = u.shape
    ky, kx = max(1, round(ny / target)), max(1, round(nx / target))
    xs, ys = lon[::kx], lat[::ky]
    uu, vv = u[::ky, ::kx], v[::ky, ::kx]
    X, Y = np.meshgrid(xs, ys)
    sp = np.hypot(uu, vv)
    ok = np.isfinite(sp) & (sp > 0)
    if not ok.any():
        return None
    c = math.cos(math.radians(lat0))
    dlat = abs(float(np.median(np.diff(ys)))) if len(ys) > 1 else 1.0
    dlon = abs(float(np.median(np.diff(xs)))) * c if len(xs) > 1 else 1.0
    lmax = 0.85 * min(dlat, dlon)
    ref = float(np.nanpercentile(sp[ok], 95)) or 1.0
    L = lmax * np.clip(sp / ref, 0.3, 1.0)
    th = np.arctan2(vv, uu)
    X0, Y0, L, th = X[ok], Y[ok], L[ok], th[ok]
    sx, sy = L * np.cos(th), L * np.sin(th)  # espaço de tela (em graus de latitude)
    tail_x, tail_y = X0 - sx / 2 / c, Y0 - sy / 2
    tip_x, tip_y = X0 + sx / 2 / c, Y0 + sy / 2
    head = 0.34 * L
    a1, a2 = th + math.radians(155), th - math.radians(155)
    h1x, h1y = tip_x + head * np.cos(a1) / c, tip_y + head * np.sin(a1)
    h2x, h2y = tip_x + head * np.cos(a2) / c, tip_y + head * np.sin(a2)
    nan = np.full(len(X0), np.nan)
    px = np.column_stack([tail_x, tip_x, h1x, nan, tip_x, h2x, nan]).ravel()
    py = np.column_stack([tail_y, tip_y, h1y, nan, tip_y, h2y, nan]).ravel()
    return px, py


# --------------------------------------------------------------------------
# Figura
# --------------------------------------------------------------------------
def build_map_figure(data, *, title, subtitle="", colorscale=None, theme="Escuro", style="Suave",
                     opacity=0.94, borders=True, labels=True, land=True, robust=True,
                     point=None, point_label="Ponto", height=None):
    th = THEMES.get(theme, THEMES["Escuro"])
    key, units = data["key"], data.get("units", "")
    lat, lon = np.asarray(data["lat"], float), np.asarray(data["lon"], float)
    field = np.asarray(data["field"], float)
    u = np.asarray(data["u"], float) if "u" in data else None
    v = np.asarray(data["v"], float) if "v" in data else None

    if lat.ndim == 2:  # curvilínea -> regular
        try:
            arrs = [field] + ([u, v] if u is not None else [])
            lon1, lat1, outs = _regrid(lat, lon, arrs)
            field = outs[0]
            if u is not None:
                u, v = outs[1], outs[2]
            lon, lat = lon1, lat1
        except ImportError:  # sem scipy: pontos
            lon, lat = lon.ravel(), lat.ravel()
            field = field.ravel()
            u = v = None
    if lat.ndim == 1 and field.ndim == 2:
        if lat[0] > lat[-1]:
            lat, field = lat[::-1], field[::-1]
            u, v = (u[::-1], v[::-1]) if u is not None else (None, None)
        if lon[0] > lon[-1]:
            lon, field = lon[::-1], field[:, ::-1]
            u, v = (u[:, ::-1], v[:, ::-1]) if u is not None else (None, None)
    if _is_precip_key(key) and units in ("mm", "mm/h"):
        field = np.where(field <= 0.05, np.nan, field)

    fin = field[np.isfinite(field)]
    zmin = zmax = None
    if fin.size:
        zmin, zmax = (np.percentile(fin, [1, 99]) if robust else (fin.min(), fin.max()))
        if not zmax > zmin:
            zmin, zmax = float(fin.min()), float(fin.max()) + 1e-9
    cs = colorscale if colorscale is not None else default_colorscale(key)

    x0, x1 = float(np.nanmin(lon)), float(np.nanmax(lon))
    y0, y1 = float(np.nanmin(lat)), float(np.nanmax(lat))
    pad = 0.0
    bbox = (y0 - pad, y1 + pad, x0 - pad, x1 + pad)
    lat0 = max(min((y0 + y1) / 2, 70), -70)
    res = _resolution(bbox)

    fig = go.Figure()
    if land:  # shape "below": heatmaps ficam acima; a terra só aparece onde o campo é vazio
        path = _land_path(bbox, res)
        if path:
            fig.add_shape(type="path", path=path, xref="x", yref="y", fillcolor=th["land"],
                          line=dict(width=0), layer="below")
    cbar = dict(title=dict(text=units, side="right", font=dict(color=th["muted"], size=12)), orientation="h",
                thickness=11, len=0.45, lenmode="fraction", x=0.5, xanchor="center", y=-0.03, yanchor="top",
                outlinewidth=0, tickfont=dict(color=th["muted"], size=11))
    hover = "%{y:.2f}°, %{x:.2f}°<br><b>%{z:.3g}</b> " + (units or "") + "<extra></extra>"
    if field.ndim == 2:
        if style == "Contornos":
            fig.add_trace(go.Contour(x=lon, y=lat, z=field, colorscale=cs, zmin=zmin, zmax=zmax, ncontours=20,
                                     contours=dict(coloring="fill", showlines=False), line=dict(width=0),
                                     opacity=opacity, colorbar=cbar, hovertemplate=hover, connectgaps=False))
        else:
            fig.add_trace(go.Heatmap(x=lon, y=lat, z=field, colorscale=cs, zmin=zmin, zmax=zmax,
                                     zsmooth="best" if style == "Suave" else False, opacity=opacity,
                                     colorbar=cbar, hovertemplate=hover, hoverongaps=False))
    else:
        fig.add_trace(go.Scattergl(x=lon, y=lat, mode="markers", marker=dict(
            symbol="square", size=6, color=field, colorscale=cs, cmin=zmin, cmax=zmax, colorbar=cbar),
            hovertemplate=hover.replace("%{z", "%{marker.color"), showlegend=False))
    if borders:
        bx, by = _polyline(bbox, res)
        if bx.size:  # halo + linha: legível sobre qualquer paleta; mais fino em mapas amplos
            span = max(x1 - x0, y1 - y0)
            wh, wb = (1.4, 0.55) if span > 120 else (2.0, 0.75) if span > 40 else (2.6, 0.9)
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=th["halo"], width=wh),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=th["border"], width=wb),
                                     hoverinfo="skip", showlegend=False))
    if labels:
        lx, ly, names = _labels(bbox, res)
        if names:
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=names, hoverinfo="skip", showlegend=False,
                                     textfont=dict(color=th["label"], size=10)))
    if u is not None and field.ndim == 2:
        arr = _arrows(lon, lat, u, v, lat0)
        if arr is not None:
            fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode="lines", line=dict(color=th["arrow"], width=1.1),
                                     hoverinfo="skip", showlegend=False))
    if point and x0 <= point[1] <= x1 and y0 <= point[0] <= y1:
        fig.add_trace(go.Scatter(x=[point[1]], y=[point[0]], mode="markers+text", text=[point_label],
                                 textposition="top center", textfont=dict(color=th["text"], size=11),
                                 marker=dict(symbol="circle", size=9, color=th["point"],
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate=f"{point_label}: {point[0]:.2f}°, {point[1]:.2f}°<extra></extra>",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[point[1]], y=[point[0]], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(symbol="circle-open", size=20, line=dict(color=th["point"], width=2))))

    axis = dict(showgrid=True, gridcolor=th["grid"], zeroline=False, showline=False,
                tickfont=dict(color=th["muted"], size=11), ticksuffix="°")
    fig.update_xaxes(range=[x0, x1], constrain="domain", **axis)
    fig.update_yaxes(range=[y0, y1], scaleanchor="x", scaleratio=1 / math.cos(math.radians(lat0)),
                     constrain="domain", **axis)
    if height is None:  # proporção do mapa (largura de referência ~1100 px) + título e barra de cores
        ratio = (y1 - y0) / math.cos(math.radians(lat0)) / max(x1 - x0, 1e-6)
        height = int(min(max(1000 * ratio + 190, 460), 800))
    sub = f"<br><span style='font-size:12px;color:{th['muted']}'>{subtitle}</span>" if subtitle else ""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>{sub}", x=0.01, xanchor="left", y=0.97,
                   font=dict(color=th["text"], size=17)),
        paper_bgcolor=th["paper"], plot_bgcolor=th["plot"], height=height,
        margin=dict(l=8, r=8, t=78, b=64), dragmode="pan", hoverlabel=dict(font_size=12),
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color=th["text"]))
    return fig
