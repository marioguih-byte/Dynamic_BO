# -*- coding: utf-8 -*-
"""
Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente TODOS os datasets do catálogo (NOAA, ECMWF, Google WeatherNext,
DWD, NASA, UCSB, ECCC...) e TODAS as variáveis de cada um. Horário sempre em Brasília.

    pip install -r requirements.txt
    streamlit run app.py
"""
import inspect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.colors import qualitative as Q
from plotly.subplots import make_subplots

import dynamical_core as C
import mapview as MV
import maplibre_view as ML

st.set_page_config(page_title="Previsão · dynamical.org", page_icon="🌦️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:4.5rem;max-width:1400px}
[data-testid="stExpander"]{border-radius:12px}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
.muted{color:#8d99ae;font-size:.85rem}
</style>""", unsafe_allow_html=True)

OTHER = "Outro local…"
CITIES = {
    "Brasília": (-15.78, -47.93), "Buenos Aires": (-34.60, -58.38), "Cairo": (30.04, 31.24),
    "Cidade do México": (19.43, -99.13), "Joanesburgo": (-26.20, 28.05), "Lagos": (6.52, 3.38),
    "Londres": (51.51, -0.13), "Moscou": (55.76, 37.62), "Nairóbi": (-1.29, 36.82),
    "Nova Délhi": (28.61, 77.21), "Nova York": (40.71, -74.01), "Pequim": (39.90, 116.41),
    "São Paulo": (-23.55, -46.63), "Singapura": (1.35, 103.82), "Sydney": (-33.87, 151.21),
    "Tóquio": (35.68, 139.69), OTHER: None,
}
REGIONS = {  # (S, N, W, E) aproximadas
    "Global": (-90, 90, -180, 180),
    "América do Norte": (5, 75, -170, -50), "América do Sul": (-56, 14, -85, -32),
    "Europa": (34, 72, -25, 45), "África": (-36, 38, -20, 55), "Ásia": (-10, 75, 25, 150),
    "Oceania": (-50, 0, 110, 180), "EUA (CONUS)": (24, 50, -125, -66), "Brasil": (-35, 6, -75, -33),
    "Personalizada": None,
}
QUALITY = {"Rápida (≈0,75°)": 150_000, "Padrão (≈0,5°)": 600_000, "Alta (nativa)": 1_500_000}
RED_PT = {"mean": "média", "std": "dispersão", "max": "máximo", "min": "mínimo", "member": "membro"}
MAP_SIZE = 720  # lado do painel quadrado do mapa interativo (px)
VIEWS = ["🧭 Mapa interativo", "📈 Mapa Plotly"]
DEFAULT_MODELS = ["noaa-gfs-forecast", "ecmwf-aifs-single-forecast", "ecmwf-ifs-ens-forecast-15-day-0-25-degree"]
DEFAULT_VARS = ["temperature_2m", "wind_speed_10m", "precipitation_surface__acum"]
PALETTE = list(Q.Plotly) + list(Q.Dark24)
LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200]

ss = st.session_state


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def stretch(fn):
    """Kwarg de largura total compatível com várias versões do Streamlit."""
    params = inspect.signature(fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


@st.cache_data(ttl=3600, show_spinner="Consultando o catálogo do dynamical.org…")
def catalog_ids():
    return C.list_dataset_ids()


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
    return hex_color


def option_labels(items, n_models=None):
    """Rótulos em português '<nome> (<unidade>)'; desambigua duplicados e informa cobertura parcial."""
    base = {}
    for v in items:
        u = C.unit_label(v["key"], v["units"], ss.wind_unit)
        base[v["key"]] = f"{v['label']} ({u})" if u else v["label"]
    dup = {l for l, n in Counter(base.values()).items() if n > 1}
    out = {}
    for v in items:
        lab = base[v["key"]] + (f" — {v['key']}" if base[v["key"]] in dup else "")
        if n_models and v["count"] < n_models:
            lab += f" · {v['count']}/{n_models} modelos"
        out[v["key"]] = lab
    return out


# estado inicial
ss.setdefault("city", "Brasília")
ss.setdefault("lat", CITIES["Brasília"][0])
ss.setdefault("lon", CITIES["Brasília"][1])
ss.setdefault("wind_unit", "m/s")
ss.setdefault("workers", 4)
ss.setdefault("ens_mode", "faixa P10–P90")
ss.setdefault("horizon", 7)
ss.setdefault("adays", 3)
ss.setdefault("skip", True)
for k, v in zip(("bb_s", "bb_n", "bb_w", "bb_e"), REGIONS["Global"]):
    ss.setdefault(k, float(v))
ss.setdefault("map_region", "Global")
ss.setdefault("map_quality", "Padrão (≈0,5°)")
ss.setdefault("map_theme", "Escuro")
ss.setdefault("map_style", "Suave")
ss.setdefault("map_palette", "Automática")
ss.setdefault("map_view", VIEWS[0])
ss.setdefault("map_base", ML.DEFAULT_BASEMAP)
ss.setdefault("map_arrows", True)
ss.setdefault("map_opacity", 0.9)
for _k in ("map_robust", "map_borders", "map_labels", "map_land"):
    ss.setdefault(_k, True)


def _on_city():
    c = CITIES.get(ss.city)
    if c:
        ss.lat, ss.lon = c


def _on_region():
    box = REGIONS.get(ss.map_region)
    if box:
        ss.bb_s, ss.bb_n, ss.bb_w, ss.bb_e = [float(x) for x in box]


def _on_bbox():
    ss.map_region = "Personalizada"


# --------------------------------------------------------------------------
# barra lateral (mínima)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌦️ Previsão")
    st.caption("dynamical.org · todos os modelos e variáveis")
    st.markdown(f"🕒 **Agora em Brasília:** {C.now_brasilia():%d/%m %H:%M}")
    st.caption("Todos os horários em Brasília · sempre a rodada mais recente de cada modelo.")
    st.selectbox("Vento em", list(C.WIND_FACTORS), key="wind_unit")
    with st.expander("Avançado"):
        st.slider("Conexões paralelas", 1, 8, key="workers")
        if st.button("🔄 Recarregar catálogo e caches"):
            st.cache_data.clear()
            C.clear_caches()
            for k in ("ts", "map_res", "inspect_all"):
                ss.pop(k, None)
            st.rerun()

LAT, LON = float(ss.lat), float(ss.lon)
tab_ts, tab_map, tab_cat = st.tabs(["📈 Série temporal", "🗺️ Mapa", "📚 Catálogo"])
IDS, SRC = catalog_ids()


# --------------------------------------------------------------------------
# Série temporal
# --------------------------------------------------------------------------
def _preset_models(kind):
    ids, _ = catalog_ids()
    if kind == "fast":
        sel = [i for i in ids if not C.is_virtual(i) and not C.is_analysis_id(i) and C.coverage_of(i)[0] == "Global"]
    elif kind == "cover":
        sel = [i for i in ids if not C.is_analysis_id(i) and C.covers_point(i, float(ss.lat), float(ss.lon))]
    elif kind == "forecast":
        sel = [i for i in ids if not C.is_analysis_id(i)]
    elif kind == "all":
        sel = list(ids)
    else:
        sel = []
    ss.sel_models = sel


def _preset_vars(kind, options):
    if kind == "all":
        ss.sel_vars = list(options)
    elif kind == "default":
        ss.sel_vars = [v for v in DEFAULT_VARS if v in options]
    else:
        ss.sel_vars = []


def build_series_fig(ts):
    results, order = ts["results"], ts["choices"]
    wind = ss.wind_unit
    keys = []
    for res in results.values():
        for k in res["vars"]:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: order.index(k.split("@")[0]) if k.split("@")[0] in order else 999)
    titles = []
    for k in keys:
        meta = next(r["vars"][k] for r in results.values() if k in r["vars"])
        _, unit = C.display_values(meta, meta["data"][..., :1], wind)
        titles.append(f"{meta['label']}" + (f"  [{unit}]" if unit else ""))
    n = len(keys)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=min(0.06, 0.8 / max(n - 1, 1)))
    shown, lo_x, hi_x = set(), None, None
    for di, (did, res) in enumerate(results.items()):
        color = PALETTE[di % len(PALETTE)]
        name = C.pretty(did) + (f" · rodada {C.to_brasilia(res['init']):%d/%m %Hh}" if res["init"] is not None else "")
        for ri, k in enumerate(keys, 1):
            meta = res["vars"].get(k)
            if not meta:
                continue
            arr, _ = C.display_values(meta, meta["data"], wind)
            mean, lo, hi = C.summarize(arr)
            x = C.to_brasilia(meta.get("valid", res["valid"]))
            lo_x = x[0] if lo_x is None else min(lo_x, x[0])
            hi_x = x[-1] if hi_x is None else max(hi_x, x[-1])
            if arr.ndim == 2 and ss.ens_mode == "membros":
                for m in arr:
                    fig.add_trace(go.Scatter(x=x, y=m, mode="lines", line=dict(color=color, width=0.6),
                                             opacity=0.2, legendgroup=did, showlegend=False, hoverinfo="skip"),
                                  row=ri, col=1)
            elif lo is not None:
                fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), legendgroup=did,
                                         showlegend=False, hoverinfo="skip"), row=ri, col=1)
                fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                         fillcolor=rgba(color, 0.15), legendgroup=did, showlegend=False,
                                         hoverinfo="skip"), row=ri, col=1)
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=name, legendgroup=did,
                                     showlegend=did not in shown, line=dict(color=color, width=2)),
                          row=ri, col=1)
            shown.add(did)
    now = C.now_brasilia()
    if lo_x is not None:
        fig.update_xaxes(range=[lo_x, hi_x])
        if lo_x <= now <= hi_x:  # marca o instante atual em todos os painéis
            for r in range(1, n + 1):
                fig.add_shape(type="line", x0=now, x1=now, y0=0, y1=1, xref="x" if r == 1 else f"x{r}",
                              yref="y domain" if r == 1 else f"y{r} domain",
                              line=dict(color="rgba(150,150,150,0.8)", width=1, dash="dot"))
            fig.add_annotation(x=now, y=1, xref="x", yref="y domain", text="agora", showarrow=False,
                               yanchor="bottom", font=dict(size=11, color="#8d99ae"))
    fig.update_layout(height=max(460, 250 * n), hovermode="x unified", margin=dict(t=70, l=10, r=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=True, hoverformat="%a %d/%m %Hh")
    fig.update_xaxes(title_text="Horário de Brasília", row=n, col=1)
    return fig


def run_series(sel, choices, horizon, adays, levels, skip_nocover):
    todo = [i for i in sel if not (skip_nocover and not C.covers_point(i, LAT, LON))]
    errors = {i: "fora da cobertura (heurística) – desmarque 'pular' em Mais opções para tentar mesmo assim"
              for i in sel if i not in todo}
    results = {}
    if todo:
        prog = st.progress(0.0, text="Baixando…")
        with ThreadPoolExecutor(max_workers=ss.workers) as ex:
            futs = {ex.submit(C.fetch_series, i, LAT, LON, choices, init_req=None, horizon_days=horizon,
                              analysis_days=adays, levels=levels): i for i in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa
                    errors[i] = f"{type(exc).__name__}: {exc}"
                prog.progress(n / len(todo), text=f"{n}/{len(todo)} · {C.pretty(i)}")
        prog.empty()
    ss.ts = {"results": {i: results[i] for i in sel if i in results}, "errors": errors, "choices": choices}


with tab_ts:
    # --- local
    p1, p2, p3 = st.columns([2, 1, 1])
    p1.selectbox("📍 Local", list(CITIES), key="city", on_change=_on_city)
    if ss.city == OTHER:
        p2.number_input("Latitude", -90.0, 90.0, step=0.25, format="%.2f", key="lat")
        p3.number_input("Longitude", -180.0, 180.0, step=0.25, format="%.2f", key="lon")

    # --- modelos
    ss.sel_models = [m for m in ss.get("sel_models", [d for d in DEFAULT_MODELS if d in IDS]) if m in IDS]
    st.multiselect("Modelos", IDS, key="sel_models", format_func=C.pretty,
                   placeholder="Escolha um ou mais modelos")
    sel = list(ss.sel_models)

    # --- variáveis (dependem dos modelos; subgrupos/níveis vêm de 'Mais opções')
    use_groups = bool(ss.get("use_groups", False))
    levels = ss.get("levels", [850, 500]) or [850, 500]
    union, options, labels, bad = [], [], {}, {}
    if sel:
        with st.spinner(f"Lendo variáveis de {len(sel)} modelo(s)…"):
            infos = C.inspect_many(sel, workers=ss.workers)
        okinfos = {k: v for k, v in infos.items() if v.get("ok")}
        bad = {k: v for k, v in infos.items() if not v.get("ok")}
        union = C.union_vars(okinfos)
        ginf = {}
        if use_groups:
            with st.spinner("Procurando subgrupos…"):
                ginf = C.inspect_groups(sel, workers=ss.workers)
            for g, dd in ginf.items():
                union += C.union_vars(dd, group=g)
        options = [v["key"] for v in union]
        labels = option_labels(union, len(sel))
    ss.sel_vars = [v for v in ss.get("sel_vars", [d for d in DEFAULT_VARS if d in options]) if v in options]
    st.multiselect("Variáveis", options, key="sel_vars", format_func=lambda k: labels.get(k, k),
                   placeholder="Escolha uma ou mais variáveis", disabled=not sel,
                   help="Vento e chuva acumulada são calculados a partir das variáveis originais.")
    choices = list(ss.sel_vars)
    far = [C.pretty(i) for i in sel if not C.covers_point(i, LAT, LON)]
    if far:
        st.caption(f"⚠️ Provavelmente fora do domínio neste local (serão pulados): {', '.join(far)}.")
    run_clicked = st.button("▶️ Buscar e plotar", type="primary", disabled=not (sel and choices))

    # --- mais opções (recolhido)
    with st.expander("⚙️ Mais opções"):
        st.caption("Atalhos de modelos")
        b = st.columns(5)
        b[0].button("⚡ Globais rápidos", on_click=_preset_models, args=("fast",), **stretch(st.button))
        b[1].button("📍 Cobrem o local", on_click=_preset_models, args=("cover",), **stretch(st.button))
        b[2].button("🌐 Todas as previsões", on_click=_preset_models, args=("forecast",), **stretch(st.button))
        b[3].button("📦 TODOS", on_click=_preset_models, args=("all",), **stretch(st.button))
        b[4].button("🧹 Limpar", on_click=_preset_models, args=("none",), **stretch(st.button))
        st.caption("Atalhos de variáveis")
        v = st.columns(5)
        v[0].button("Padrão", on_click=_preset_vars, args=("default", options), **stretch(st.button))
        v[1].button("TODAS as variáveis", on_click=_preset_vars, args=("all", options), **stretch(st.button))
        v[2].button("Limpar variáveis", on_click=_preset_vars, args=("none", options), **stretch(st.button))
        o1, o2, o3 = st.columns(3)
        o1.slider("Horizonte (dias)", 1, 35, key="horizon")
        o2.slider("Histórico p/ modelos de análise (dias)", 1, 30, key="adays")
        o3.radio("Conjuntos (ensembles)", ["faixa P10–P90", "membros"], key="ens_mode")
        o4, o5, o6 = st.columns(3)
        o4.checkbox("Pular modelos fora da cobertura", key="skip")
        o5.checkbox("Incluir subgrupos (ex.: níveis de pressão)", key="use_groups")
        o6.multiselect("Níveis (hPa)", LEVELS, default=[850, 500], key="levels", disabled=not ss.get("use_groups"))
        if len(choices) > 12:
            st.caption(f"⚠️ {len(choices)} variáveis = {len(choices)} painéis; o gráfico ficará bem alto.")

    if bad:
        with st.expander(f"⚠️ {len(bad)} modelo(s) não puderam ser abertos"):
            for k, vv in bad.items():
                st.write(f"**{C.pretty(k)}** — {vv.get('error')}")
                if st.button("🩺 Diagnosticar estrutura", key=f"diag_{k}"):
                    st.code(C.structure_report(k), language="text")
            st.caption("Se a raiz do modelo não tem grade, ative 'Incluir subgrupos' em Mais opções.")

    if run_clicked:
        run_series(sel, choices, ss.horizon, ss.adays, list(ss.levels) if ss.use_groups else [850, 500], ss.skip)

    # --- resultado
    ts = ss.get("ts")
    if ts:
        res, errs = ts["results"], ts["errors"]
        if not res:
            st.error("Nenhum modelo retornou dados. Veja os detalhes abaixo.")
        else:
            st.plotly_chart(build_series_fig(ts), **stretch(st.plotly_chart))
            df = C.results_to_dataframe(res, None, ss.wind_unit)
            st.download_button("⬇️ Baixar CSV (horário de Brasília)", df.to_csv(index=False).encode("utf-8"),
                               file_name="previsao_multimodelo.csv", mime="text/csv")
        rows = []
        for i, r in res.items():
            rows.append({"modelo": C.pretty(i), "status": "ok",
                         "rodada (Brasília)": f"{C.to_brasilia(r['init']):%d/%m %H:%M}" if r["init"] is not None else "análise",
                         "grade (lat, lon)": f"{r['grid'][0]:.2f}, {r['grid'][1]:.2f}", "variáveis": len(r["vars"]),
                         "ausentes": ", ".join(r["skipped"]) or "", "obs": "; ".join(r.get("notes", []))})
        for i, e in errs.items():
            rows.append({"modelo": C.pretty(i), "status": f"✘ {e}", "rodada (Brasília)": "", "grade (lat, lon)": "",
                         "variáveis": 0, "ausentes": "", "obs": ""})
        with st.expander(f"Detalhes por modelo ({len(res)} ok · {len(errs)} falhas/pulados)", expanded=bool(errs)):
            st.dataframe(pd.DataFrame(rows), hide_index=True, **stretch(st.dataframe))


# --------------------------------------------------------------------------
# Mapa
# --------------------------------------------------------------------------
def run_map(mid, choice, bbox, kw):
    with st.spinner("Baixando o campo… (regiões grandes e conjuntos demoram)"):
        try:
            data = C.fetch_map(mid, choice, bbox, **kw)
        except Exception as exc:  # noqa
            ss.map_res = {"error": f"{type(exc).__name__}: {exc}"}
            return
    ss.map_res = {"data": data, "mid": mid, "bbox": bbox}


with tab_map:
    m1, m2 = st.columns(2)
    mid = m1.selectbox("Modelo", IDS, format_func=C.pretty, key="map_ds",
                       index=IDS.index("noaa-gfs-forecast") if "noaa-gfs-forecast" in IDS else 0)
    with st.spinner("Lendo metadados…"):
        root_info = C.inspect_dataset(mid)
    root_ok = bool(root_info.get("ok"))
    use_g = bool(ss.get("map_groups", False))
    groups_info = {}
    if use_g or not root_ok:
        for g in C.list_groups(mid):
            gi = C.inspect_dataset(mid, g)
            if gi.get("ok"):
                groups_info[g] = gi
    minfo = root_info if root_ok else next(iter(groups_info.values()), None)
    run_clicked = False
    kw = {"init_req": None, "wind_unit": ss.wind_unit}

    if minfo is None:
        st.error(f"Não foi possível abrir este modelo: {root_info.get('error')}")
        if st.button("🩺 Diagnosticar estrutura", key="diag_map_err"):
            st.code(C.structure_report(mid), language="text")
    else:
        if not root_ok:
            st.warning("A raiz deste modelo não tem grade lat/lon reconhecível; usando as variáveis dos subgrupos: "
                       + ", ".join(f"`{g}`" for g in groups_info))
        mvars = C.union_vars({mid: root_info}) if root_ok else []
        for g, gi in groups_info.items():
            mvars += C.union_vars({mid: gi}, group=g)
        mvars = [x for x in mvars if not x["key"].split("/")[-1].startswith("wind_direction_")]
        mopts = [x["key"] for x in mvars]
        mfmt = option_labels(mvars)
        default_var = mopts.index("temperature_2m") if "temperature_2m" in mopts else 0
        choice = m2.selectbox("Variável", mopts, index=default_var, format_func=lambda k: mfmt.get(k, k),
                              key=f"map_var_{mid}")

        # --- horário (Brasília): por padrão o mais próximo de agora
        t1, t2, t3 = st.columns([3, 1.4, 1])
        if minfo["kind"] == "forecast":
            init0 = pd.Timestamp(minfo["t1"])
            opts = [init0 + pd.Timedelta(hours=h) for h in minfo["leads_h"]]
            now_utc = pd.Timestamp.now("UTC").tz_localize(None)
            near = min(range(len(opts)), key=lambda i: abs(opts[i] - now_utc))
            picked = t1.select_slider("🕒 Horário (Brasília)", options=opts, value=opts[near],
                                      format_func=C.br_label, key=f"valid_{mid}_{minfo['t1']}",
                                      help="Padrão: o horário mais próximo de agora. Para chuva acumulada: "
                                           "soma desde o início da rodada até este horário.")
            kw["valid_req"] = picked.to_datetime64()
            kw["lead_h"] = float((picked - init0) / pd.Timedelta(hours=1))
        else:
            t1.caption("🕒 Instante mais recente disponível (mude em Mais opções).")
        t2.selectbox("Região", list(REGIONS), key="map_region", on_change=_on_region)
        t3.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        run_clicked = t3.button("🗺️ Gerar mapa", type="primary", **stretch(st.button))

        # --- mais opções (recolhido)
        with st.expander("⚙️ Mais opções"):
            e1, e2, e3 = st.columns(3)
            if minfo["ens"]:
                modes = {"média dos membros": "mean", "dispersão (desvio-padrão)": "std", "máximo": "max",
                         "mínimo": "min", "membro específico": "member"}
                lab = e1.selectbox("Conjunto", list(modes), key=f"mmode_{mid}")
                kw["member_mode"] = modes[lab]
                if kw["member_mode"] == "member":
                    kw["member_idx"] = e1.number_input("Membro", 0, max(minfo["members"] - 1, 0), 0, key=f"mmem_{mid}")
            if minfo["kind"] == "analysis":
                when = e1.radio("Instante", ["Mais recente", "Escolher"], horizontal=True, key=f"when_{mid}")
                if when == "Escolher":
                    d = e1.date_input("Data (UTC)", value=pd.Timestamp(minfo["t1"]).date(), key=f"mdate_{mid}")
                    h = e1.number_input("Hora (UTC)", 0, 23, 0, key=f"mhour_{mid}")
                    kw["time_req"] = (pd.Timestamp(d) + pd.Timedelta(hours=int(h))).to_datetime64()
            base_name = choice.split("/")[-1]
            cg = choice.split("/")[0] if "/" in choice else None
            vsrc = groups_info[cg]["vars"] if cg in groups_info else (root_info["vars"] if root_ok else [])
            vinfo = {x["name"]: x for x in vsrc}
            bname = base_name if base_name in vinfo else (
                "wind_u_" + base_name.split("wind_speed_")[-1] if base_name.startswith("wind_speed_") else base_name)
            vi = vinfo.get(bname)
            if vi and vi["extra"]:
                d0 = vi["extra"][0]
                vals = vi["extra_vals"].get(d0, [])
                if vals and all(isinstance(x, float) for x in vals):
                    hp = [round(x / 100.0, 1) if max(vals) > 2000 else x for x in vals]
                    kw["level"] = e2.selectbox("Nível (hPa)", hp, index=hp.index(850.0) if 850.0 in hp else 0,
                                               key=f"lvl_{mid}_{bname}")
            e3.selectbox("Resolução", list(QUALITY), key="map_quality",
                         help="Limita as células por passo. 'Alta' = grade nativa (global envia ~10 MB ao navegador).")
            kw["max_cells"] = QUALITY[ss.map_quality]

            f1, f2, f3, f4 = st.columns(4)
            f1.selectbox("Paleta", list(MV.PALETTES), key="map_palette")
            f2.selectbox("Mapa-base", list(ML.BASEMAPS), key="map_base",
                         help="Escuro/Claro: mapa próprio, funciona em qualquer navegador e sem internet. "
                              "Online: ruas e rótulos do OpenFreeMap; se não carregar, o painel volta sozinho ao modo offline.")
            f3.slider("Opacidade", 0.3, 1.0, step=0.05, key="map_opacity")
            f4.checkbox("Escala robusta (P1–P99)", key="map_robust", help="Ignora valores extremos.")
            g1, g2, g3 = st.columns(3)
            g1.selectbox("Estilo do campo", ["Suave", "Pixels", "Contornos"], key="map_style")
            g2.checkbox("Setas de vento", key="map_arrows")
            g3.radio("Tipo de mapa", VIEWS, key="map_view")
            if ss.map_view == VIEWS[1]:
                h1, h2, h3, h4 = st.columns(4)
                h1.selectbox("Tema (Plotly)", list(MV.THEMES), key="map_theme")
                h2.checkbox("Contorno dos países", key="map_borders")
                h3.checkbox("Nomes dos países", key="map_labels")
                h4.checkbox("Terra de fundo", key="map_land")
            st.caption("Limites da região (S, N, O, L)")
            cc = st.columns(4)
            cc[0].number_input("Sul", -90.0, 90.0, key="bb_s", step=1.0, on_change=_on_bbox)
            cc[1].number_input("Norte", -90.0, 90.0, key="bb_n", step=1.0, on_change=_on_bbox)
            cc[2].number_input("Oeste", -180.0, 180.0, key="bb_w", step=1.0, on_change=_on_bbox)
            cc[3].number_input("Leste", -180.0, 180.0, key="bb_e", step=1.0, on_change=_on_bbox)
            i1, i2 = st.columns(2)
            i1.checkbox("Incluir subgrupos (ex.: níveis de pressão)", key="map_groups")
            if i2.button("🩺 Diagnosticar estrutura do modelo", key="diag_map"):
                st.code(C.structure_report(mid), language="text")
        bbox = (min(ss.bb_s, ss.bb_n), max(ss.bb_s, ss.bb_n), min(ss.bb_w, ss.bb_e), max(ss.bb_w, ss.bb_e))
        if run_clicked:
            run_map(mid, choice, bbox, kw)

    # --- resultado
    mr = ss.get("map_res")
    if mr:
        if "error" in mr:
            st.error(mr["error"])
        else:
            d = mr["data"]
            red = "" if d["members"] == 1 else f" · {RED_PT.get(d['reduction'], d['reduction'])} de {d['members']} membros"
            valid_txt = f"{C.br_full(d['valid'])} (Brasília)"
            sub = valid_txt + (f" · rodada {C.br_label(d['init'])}" if d["init"] is not None else "") + red
            title = f"{C.pretty(mr['mid'])} · {d['label']}"
            f = d["field"]
            view = ss.map_view
            html = None
            if view == VIEWS[0]:
                try:
                    html, ginfo = ML.build_html(
                        d, title=title, subtitle=sub, colorscale=MV.PALETTES.get(ss.map_palette),
                        basemap=ss.map_base, opacity=ss.map_opacity,
                        resampling="nearest" if ss.map_style == "Pixels" else "linear",
                        robust=ss.map_robust, show_arrows=ss.map_arrows, size=MAP_SIZE)
                except Exception as exc:  # noqa
                    st.error(f"Não foi possível montar o mapa interativo ({type(exc).__name__}: {exc}). "
                             "Mostrando a visão Plotly.")
            if html is not None:
                left, right = st.columns([3, 1])
                with left:
                    components.html(html, height=MAP_SIZE + 8)
                with right:
                    st.markdown(f"**{C.pretty(mr['mid'])}**")
                    st.markdown(f"{d['label']} · {d['units']}" if d["units"] else d["label"])
                    st.markdown(f"<span class='muted'>🕒 {valid_txt}</span>", unsafe_allow_html=True)
                    if d["init"] is not None:
                        st.markdown(f"<span class='muted'>📌 rodada {C.br_label(d['init'])}{red}</span>",
                                    unsafe_allow_html=True)
                    st.markdown(f"Máx **{np.nanmax(f):.3g}** · Média **{np.nanmean(f):.3g}** · Mín **{np.nanmin(f):.3g}**")
                    st.download_button("⬇️ Baixar mapa (HTML)", html.encode("utf-8"),
                                       file_name="mapa_previsao.html", mime="text/html")
                st.caption("Arraste para mover · role para dar zoom · clique para ver o valor.")
            else:
                fig = MV.build_map_figure(
                    d, title=title, subtitle=sub,
                    colorscale=MV.PALETTES.get(ss.map_palette), theme=ss.map_theme, style=ss.map_style,
                    opacity=ss.map_opacity, borders=ss.map_borders, labels=ss.map_labels, land=ss.map_land,
                    robust=ss.map_robust)
                try:
                    st.plotly_chart(fig, theme=None, config=MV.PLOT_CONFIG, **stretch(st.plotly_chart))
                except TypeError:
                    st.plotly_chart(fig, config=MV.PLOT_CONFIG, **stretch(st.plotly_chart))
                st.caption(f"Mín {np.nanmin(f):.3g} · média {np.nanmean(f):.3g} · máx {np.nanmax(f):.3g} {d['units']}")
                st.download_button("⬇️ Baixar mapa (HTML)",
                                   fig.to_html(include_plotlyjs="cdn", config=MV.PLOT_CONFIG).encode("utf-8"),
                                   file_name="mapa_previsao_plotly.html", mime="text/html")
            notes = []
            if d["stride"] > 1:
                notes.append(f"grade reamostrada (a cada {d['stride']} pontos) para caber na memória")
            if d["lat"].ndim == 2 and "u" in d:
                notes.append("em grades curvilíneas as setas seguem os eixos da grade")
            if notes:
                st.caption("ℹ️ " + " · ".join(notes))


# --------------------------------------------------------------------------
# Catálogo
# --------------------------------------------------------------------------
def inspect_all(ids):
    prog = st.progress(0.0, text="Abrindo modelos…")
    infos = {}
    with ThreadPoolExecutor(max_workers=ss.workers + 2) as ex:
        futs = {ex.submit(C.inspect_dataset, i): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            infos[futs[fut]] = fut.result()
            prog.progress(n / len(ids), text=f"{n}/{len(ids)} · {C.pretty(futs[fut])}")
    prog.empty()
    ss.inspect_all = infos


with tab_cat:
    st.caption(f"{len(IDS)} modelos no catálogo · fonte: {SRC}")
    base = pd.DataFrame([{
        "modelo": i, "provedor": C.provider_of(i)[0],
        "tipo": "análise" if C.is_analysis_id(i) else "previsão",
        "otimização": "virtual (mapas, baixa latência)" if C.is_virtual(i) else "time-optimized (séries)",
        "cobertura": C.coverage_of(i)[0]} for i in IDS])
    if st.button("🔎 Inspecionar TODOS os modelos (pode levar alguns minutos)"):
        inspect_all(IDS)
    allinf = ss.get("inspect_all")
    if allinf:
        rows = []
        for i in IDS:
            v = allinf.get(i, {})
            rows.append({
                "status": "ok" if v.get("ok") else f"✘ {v.get('error', '?')[:80]}",
                "estrutura": v.get("kind", ""),
                "membros": v.get("members", "") if v.get("ens") else ("" if not v.get("ok") else 1),
                "grade": ("regular" if v.get("regular") else "curvilínea") if v.get("ok") else "",
                "lat": f"{v['lat'][0]:.1f}…{v['lat'][1]:.1f}" if v.get("ok") else "",
                "lon": f"{v['lon'][0]:.1f}…{v['lon'][1]:.1f}" if v.get("ok") else "",
                "início": v.get("t0", "")[:16], "fim": v.get("t1", "")[:16],
                "alcance máx (h)": v.get("lead_max_h", ""), "nº variáveis": len(v["vars"]) if v.get("ok") else ""})
        full = pd.concat([base, pd.DataFrame(rows)], axis=1)
        st.dataframe(full, hide_index=True, **stretch(st.dataframe))
        st.download_button("⬇️ Baixar tabela (CSV)", full.to_csv(index=False).encode("utf-8"),
                           file_name="modelos_dynamical.csv", mime="text/csv")
        names = sorted({x["name"] for v in allinf.values() if v.get("ok") for x in v["vars"]})
        mat = pd.DataFrame("", index=names, columns=IDS)
        for i, v in allinf.items():
            if v.get("ok"):
                for x in v["vars"]:
                    mat.loc[x["name"], i] = "✓"
        with st.expander(f"Matriz variável × modelo ({len(names)} variáveis distintas)"):
            st.dataframe(mat, **stretch(st.dataframe))
            st.download_button("⬇️ Baixar matriz (CSV)", mat.to_csv().encode("utf-8"),
                               file_name="matriz_variaveis.csv", mime="text/csv")
    else:
        st.dataframe(base, hide_index=True, **stretch(st.dataframe))
    st.divider()
    one = st.selectbox("Ver variáveis de um modelo", IDS, format_func=C.pretty, key="cat_one")
    with st.spinner("Abrindo…"):
        oi = C.inspect_dataset(one)

    def _vtable(vs):
        return pd.DataFrame([{"variável": v["name"], "nome": C.label_pt(v["name"], v["long_name"]),
                              "unidade original": v["units"],
                              "unidade exibida": C.unit_label(v["name"], v["units"], ss.wind_unit),
                              "dimensões": ", ".join(v["dims"])} for v in vs])

    if oi.get("ok"):
        st.dataframe(_vtable(oi["vars"]), hide_index=True, **stretch(st.dataframe))
    else:
        st.error(oi.get("error"))
    with st.spinner("Procurando subgrupos…"):
        og = C.inspect_groups([one])
    for g, dd in og.items():
        with st.expander(f"Subgrupo `{g}` ({len(dd[one]['vars'])} variáveis)"):
            st.dataframe(_vtable(dd[one]["vars"]), hide_index=True, **stretch(st.dataframe))
    if st.button("🩺 Diagnosticar estrutura deste modelo", key="diag_cat"):
        st.code(C.structure_report(one), language="text")

st.divider()
st.caption("Dados: dynamical.org (NOAA, ECMWF, Google, DWD, NASA, UCSB, ECCC) — veja licença/atribuição de cada modelo em "
           "dynamical.org/catalog · Mapas: OpenFreeMap © OpenMapTiles, dados © OpenStreetMap · Países: Natural Earth.")
