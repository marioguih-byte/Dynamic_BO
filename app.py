# -*- coding: utf-8 -*-
"""
Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente TODOS os datasets do catálogo (NOAA, ECMWF, Google WeatherNext,
DWD, NASA, UCSB, ECCC...) e TODAS as variáveis de cada um.

    pip install -r requirements.txt
    streamlit run app.py
"""
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative as Q
from plotly.subplots import make_subplots

import dynamical_core as C
import mapview as MV

st.set_page_config(page_title="Previsão multi-modelo · dynamical.org", page_icon="🌦️", layout="wide")

CITIES = {
    "Personalizado": None,
    "Brasília": (-15.78, -47.93), "Buenos Aires": (-34.60, -58.38), "Cairo": (30.04, 31.24),
    "Cidade do México": (19.43, -99.13), "Joanesburgo": (-26.20, 28.05), "Lagos": (6.52, 3.38),
    "Londres": (51.51, -0.13), "Moscou": (55.76, 37.62), "Nairóbi": (-1.29, 36.82),
    "Nova Délhi": (28.61, 77.21), "Nova York": (40.71, -74.01), "Pequim": (39.90, 116.41),
    "São Paulo": (-23.55, -46.63), "Singapura": (1.35, 103.82), "Sydney": (-33.87, 151.21),
    "Tóquio": (35.68, 139.69),
}
REGIONS = {  # (S, N, W, E) aproximadas
    "Global": (-90, 90, -180, 180),
    "América do Norte": (5, 75, -170, -50), "América do Sul": (-56, 14, -85, -32),
    "Europa": (34, 72, -25, 45), "África": (-36, 38, -20, 55), "Ásia": (-10, 75, 25, 150),
    "Oceania": (-50, 0, 110, 180), "EUA (CONUS)": (24, 50, -125, -66), "Brasil": (-35, 6, -75, -33),
}
QUALITY = {"Rápida (≈0,75°)": 150_000, "Padrão (≈0,5°)": 600_000, "Alta (nativa)": 1_500_000}
DEFAULT_MODELS = ["noaa-gfs-forecast", "ecmwf-aifs-single-forecast", "ecmwf-ifs-ens-forecast-15-day-0-25-degree"]
DEFAULT_VARS = ["temperature_2m", "wind_speed_10m", "precipitation_surface__acum"]
PALETTE = list(Q.Plotly) + list(Q.Dark24)
LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200]

ss = st.session_state


# --------------------------------------------------------------------------
# utilidades de UI
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


def init_request():
    if ss.latest:
        return None
    return (pd.Timestamp(ss.init_date) + pd.Timedelta(hours=int(ss.init_hour))).to_datetime64()


# defaults de estado
ss.setdefault("city", "Personalizado")
ss.setdefault("lat", 0.0)
ss.setdefault("lon", 0.0)
ss.setdefault("tz", -3.0)
ss.setdefault("latest", True)
ss.setdefault("init_date", pd.Timestamp.now("UTC").date())
ss.setdefault("init_hour", 0)
ss.setdefault("wind_unit", "m/s")
ss.setdefault("workers", 4)
ss.setdefault("ens_mode", "faixa P10–P90")
for k, v in zip(("bb_s", "bb_n", "bb_w", "bb_e"), REGIONS["Global"]):
    ss.setdefault(k, float(v))
ss.setdefault("map_region", "Global")
ss.setdefault("map_quality", "Padrão (≈0,5°)")
ss.setdefault("map_theme", "Escuro")
ss.setdefault("map_style", "Suave")
ss.setdefault("map_palette", "Automática")
ss.setdefault("map_opacity", 0.94)
for _k in ("map_robust", "map_borders", "map_labels", "map_land"):
    ss.setdefault(_k, True)


def _on_city():
    c = CITIES.get(ss.city)
    if c:
        ss.lat, ss.lon = c


def _on_latlon():
    ss.city = "Personalizado"


def _on_region():
    box = REGIONS.get(ss.map_region)
    if box:
        ss.bb_s, ss.bb_n, ss.bb_w, ss.bb_e = [float(x) for x in box]


# --------------------------------------------------------------------------
# barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🌦️ Previsão multi-modelo")
    st.caption("dynamical.org · todos os modelos e variáveis do catálogo")
    st.subheader("🕒 Tempo")
    st.checkbox("Usar a rodada mais recente com dados", key="latest",
                help="Em datasets de análise (sem rodada), vale o intervalo recente definido na aba Série.")
    if not ss.latest:
        st.date_input("Data da rodada (UTC)", key="init_date")
        st.number_input("Hora da rodada (UTC)", 0, 23, step=1, key="init_hour")
        st.caption("Usa a última rodada disponível menor ou igual à informada.")
    st.number_input("Fuso horário (UTC±h)", -12.0, 14.0, step=0.5, key="tz")
    st.subheader("⚙️ Opções")
    st.selectbox("Unidade do vento", list(C.WIND_FACTORS), key="wind_unit")
    st.slider("Conexões paralelas", 1, 8, key="workers")
    if st.button("🔄 Recarregar catálogo e caches"):
        st.cache_data.clear()
        C.clear_caches()
        for k in ("ts", "map_res", "inspect_all"):
            ss.pop(k, None)
        st.rerun()
    st.caption("Dados: dynamical.org (NOAA, ECMWF, Google, DWD, NASA, UCSB, ECCC). "
               "Veja licença/atribuição de cada dataset em dynamical.org/catalog.")

LAT, LON, TZ = float(ss.lat), float(ss.lon), float(ss.tz)
tab_ts, tab_map, tab_cat, tab_about = st.tabs(["📈 Série temporal", "🗺️ Mapa", "📚 Catálogo e variáveis", "ℹ️ Sobre"])
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
        titles.append(f"{meta['label']}  [{unit}]")
    n = len(keys)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=min(0.06, 0.8 / max(n - 1, 1)))
    shown = set()
    off = pd.Timedelta(hours=TZ)
    for di, (did, res) in enumerate(results.items()):
        color = PALETTE[di % len(PALETTE)]
        name = C.pretty(did) + (f" · init {res['init']:%d/%m %HZ}" if res["init"] is not None else "")
        for ri, k in enumerate(keys, 1):
            meta = res["vars"].get(k)
            if not meta:
                continue
            arr, _ = C.display_values(meta, meta["data"], wind)
            mean, lo, hi = C.summarize(arr)
            x = meta.get("valid", res["valid"]) + off
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
    fig.update_layout(height=max(460, 250 * n), hovermode="x unified", margin=dict(t=70, l=10, r=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=True)
    fig.update_xaxes(title_text=f"Hora válida (UTC{TZ:+g})", row=n, col=1)
    return fig


def run_series(sel, choices, init_req, horizon, adays, levels, skip_nocover):
    todo = [i for i in sel if not (skip_nocover and not C.covers_point(i, LAT, LON))]
    errors = {i: "fora da cobertura (heurística) – desmarque 'pular' para tentar mesmo assim"
              for i in sel if i not in todo}
    results = {}
    if todo:
        prog = st.progress(0.0, text="Baixando…")
        with ThreadPoolExecutor(max_workers=ss.workers) as ex:
            futs = {ex.submit(C.fetch_series, i, LAT, LON, choices, init_req=init_req, horizon_days=horizon,
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
    st.caption(f"{len(IDS)} datasets no catálogo · fonte: {SRC}")
    p1, p2, p3 = st.columns([2, 1, 1])
    p1.selectbox("📍 Ponto da série", list(CITIES), key="city", on_change=_on_city,
                 help="A série temporal precisa de um ponto. O mapa (aba ao lado) é global e não usa este ponto.")
    p2.number_input("Latitude", -90.0, 90.0, step=0.25, format="%.2f", key="lat", on_change=_on_latlon)
    p3.number_input("Longitude", -180.0, 180.0, step=0.25, format="%.2f", key="lon", on_change=_on_latlon)
    b = st.columns(6)
    b[0].button("⚡ Globais rápidos", on_click=_preset_models, args=("fast",), **stretch(st.button),
                help="Previsões globais time-optimized (melhores para séries em um ponto).")
    b[1].button("📍 Cobrem o ponto", on_click=_preset_models, args=("cover",), **stretch(st.button))
    b[2].button("🌐 Todas as previsões", on_click=_preset_models, args=("forecast",), **stretch(st.button))
    b[3].button("📦 TODOS", on_click=_preset_models, args=("all",), **stretch(st.button))
    b[4].button("🧹 Limpar", on_click=_preset_models, args=("none",), **stretch(st.button))
    ss.sel_models = [m for m in ss.get("sel_models", [d for d in DEFAULT_MODELS if d in IDS]) if m in IDS]
    st.multiselect("Modelos (datasets)", IDS, key="sel_models", format_func=C.pretty,
                   help="Datasets 'virtual' são otimizados para mapas: séries nesses podem demorar.")
    sel = list(ss.sel_models)
    if not sel:
        st.info("Escolha ao menos um modelo (ou use os atalhos acima).")
    else:
        far = [C.pretty(i) for i in sel if not C.covers_point(i, LAT, LON)]
        if far:
            st.warning(f"Provavelmente fora do domínio no ponto atual: {', '.join(far)}.")
        with st.spinner(f"Lendo metadados e variáveis de {len(sel)} dataset(s)…"):
            infos = C.inspect_many(sel, workers=ss.workers)
        okinfos = {k: v for k, v in infos.items() if v.get("ok")}
        bad = {k: v for k, v in infos.items() if not v.get("ok")}
        if bad:
            with st.expander(f"⚠️ {len(bad)} dataset(s) não puderam ser abertos"):
                for k, v in bad.items():
                    st.write(f"**{C.pretty(k)}** — {v.get('error')}")
        union = C.union_vars(okinfos)
        c1, c2 = st.columns([2, 1])
        use_groups = c1.checkbox("Incluir variáveis em níveis de pressão (grupo `pressure_level`, se existir)",
                                 key="use_groups")
        levels = c2.multiselect("Níveis (hPa)", LEVELS, default=[850, 500], key="levels", disabled=not use_groups)
        if use_groups:
            ginf = C.inspect_many(list(okinfos), group="pressure_level", workers=ss.workers)
            union += C.union_vars({k: v for k, v in ginf.items() if v.get("ok")}, group="pressure_level")
            ng = sum(1 for v in ginf.values() if v.get("ok"))
            c1.caption(f"{ng} de {len(okinfos)} dataset(s) têm grupo `pressure_level`.")
        options = [v["key"] for v in union]
        fmt = {v["key"]: f"{v['key']} — {v['label']} [{v['units']}] · {v['count']}/{len(sel)}" for v in union}
        ss.sel_vars = [v for v in ss.get("sel_vars", [d for d in DEFAULT_VARS if d in options]) if v in options]
        vb = st.columns(6)
        vb[0].button("Padrão", on_click=_preset_vars, args=("default", options), **stretch(st.button))
        vb[1].button("TODAS as variáveis", on_click=_preset_vars, args=("all", options), **stretch(st.button))
        vb[2].button("Limpar variáveis", on_click=_preset_vars, args=("none", options), **stretch(st.button))
        st.multiselect(f"Variáveis ({len(options)} disponíveis nos modelos escolhidos)", options,
                       key="sel_vars", format_func=lambda k: fmt.get(k, k),
                       help="'n/m' = em quantos dos modelos escolhidos a variável existe. "
                            "Variáveis derivadas (vento, precipitação acumulada) são calculadas a partir das originais.")
        o1, o2, o3, o4 = st.columns(4)
        horizon = o1.slider("Horizonte (dias)", 1, 35, 7)
        adays = o2.slider("Histórico p/ datasets de análise (dias)", 1, 30, 3)
        o3.radio("Conjuntos (ensembles)", ["faixa P10–P90", "membros"], key="ens_mode", horizontal=True)
        skip = o4.checkbox("Pular modelos fora da cobertura", value=True)
        choices = list(ss.sel_vars)
        if len(choices) > 12:
            st.caption(f"⚠️ {len(choices)} variáveis = {len(choices)} painéis; o gráfico ficará bem alto.")
        if st.button("▶️ Buscar e plotar", type="primary", disabled=not choices):
            run_series(sel, choices, init_request(), horizon, adays, levels if use_groups else [850, 500], skip)

    ts = ss.get("ts")
    if ts:
        res, errs = ts["results"], ts["errors"]
        st.divider()
        if not res:
            st.error("Nenhum modelo retornou dados. Veja os detalhes abaixo.")
        else:
            st.plotly_chart(build_series_fig(ts), **stretch(st.plotly_chart))
        rows = []
        for i, r in res.items():
            rows.append({"dataset": C.pretty(i), "status": "ok", "init (UTC)": str(r["init"] or "análise"),
                         "grade (lat, lon)": f"{r['grid'][0]:.2f}, {r['grid'][1]:.2f}", "variáveis": len(r["vars"]),
                         "ausentes": ", ".join(r["skipped"]) or "", "obs": "; ".join(r.get("notes", []))})
        for i, e in errs.items():
            rows.append({"dataset": C.pretty(i), "status": f"✘ {e}", "init (UTC)": "", "grade (lat, lon)": "",
                         "variáveis": 0, "ausentes": "", "obs": ""})
        with st.expander(f"Status por modelo ({len(res)} ok · {len(errs)} com falha/pulados)", expanded=bool(errs)):
            st.dataframe(pd.DataFrame(rows), hide_index=True, **stretch(st.dataframe))
        if res:
            df = C.results_to_dataframe(res, TZ, ss.wind_unit)
            with st.expander(f"Dados ({len(df):,} linhas)"):
                st.dataframe(df.head(2000), hide_index=True, **stretch(st.dataframe))
            st.download_button("⬇️ Baixar CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="previsao_multimodelo.csv", mime="text/csv")


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
    m1, m2 = st.columns([2, 1])
    mid = m1.selectbox("Dataset", IDS, format_func=C.pretty, key="map_ds",
                       index=IDS.index("noaa-gfs-forecast") if "noaa-gfs-forecast" in IDS else 0)
    with st.spinner("Lendo metadados…"):
        minfo = C.inspect_dataset(mid)
    if not minfo.get("ok"):
        st.error(f"Não foi possível abrir este dataset: {minfo.get('error')}")
    else:
        use_g = m2.checkbox("Variáveis em níveis de pressão (grupo `pressure_level`)", key="map_groups")
        mvars = C.union_vars({mid: minfo})
        ginfo = None
        if use_g:
            ginfo = C.inspect_dataset(mid, "pressure_level")
            if ginfo.get("ok"):
                mvars += C.union_vars({mid: ginfo}, group="pressure_level")
            else:
                m2.caption("Este dataset não tem grupo `pressure_level`.")
        mvars = [v for v in mvars if not v["key"].startswith("wind_direction_")]
        mopts = [v["key"] for v in mvars]
        mfmt = {v["key"]: f"{v['key']} — {v['label']} [{v['units']}]" for v in mvars}
        default_var = mopts.index("temperature_2m") if "temperature_2m" in mopts else 0
        mkey = f"map_var_{mid}"
        choice = st.selectbox(f"Variável ({len(mopts)} disponíveis)", mopts, index=default_var,
                              format_func=lambda k: mfmt.get(k, k), key=mkey)

        st.caption(f"{minfo['kind']} · {'conjunto de ' + str(minfo['members']) + ' membros' if minfo['ens'] else 'determinístico'}"
                   f" · grade {'regular' if minfo['regular'] else 'curvilínea'} · "
                   f"lat {minfo['lat'][0]:.1f}…{minfo['lat'][1]:.1f}, lon {minfo['lon'][0]:.1f}…{minfo['lon'][1]:.1f}"
                   f" · dados de {minfo.get('t0', '?')[:16]} a {minfo.get('t1', '?')[:16]}")
        kw = {"init_req": init_request()}
        t1, t2, t3 = st.columns(3)
        if minfo["kind"] == "forecast":
            leads = minfo["leads_h"]
            dflt = min(leads, key=lambda h: abs(h - 24))
            kw["lead_h"] = t1.select_slider("Passo de previsão", options=leads, value=dflt,
                                            format_func=lambda h: f"+{h:g} h", key=f"lead_{mid}")
            if "__acum" in choice:
                t1.caption("Acumulado de 0 até o passo escolhido.")
        elif minfo["kind"] == "analysis":
            when = t1.radio("Instante", ["Último disponível", "Escolher"], horizontal=True, key=f"when_{mid}")
            if when == "Escolher":
                d = t1.date_input("Data (UTC)", value=pd.Timestamp(minfo["t1"]).date(), key=f"mdate_{mid}")
                h = t1.number_input("Hora (UTC)", 0, 23, 0, key=f"mhour_{mid}")
                kw["time_req"] = (pd.Timestamp(d) + pd.Timedelta(hours=int(h))).to_datetime64()
        if minfo["ens"]:
            modes = {"média dos membros": "mean", "dispersão (desvio-padrão)": "std", "máximo": "max",
                     "mínimo": "min", "membro específico": "member"}
            lab = t2.selectbox("Conjunto", list(modes), key=f"mmode_{mid}")
            kw["member_mode"] = modes[lab]
            if kw["member_mode"] == "member":
                kw["member_idx"] = t2.number_input("Membro", 0, max(minfo["members"] - 1, 0), 0, key=f"mmem_{mid}")
        t3.selectbox("Paleta", list(MV.PALETTES), key="map_palette")
        # nível (quando a variável tem dimensão vertical)
        base_name = choice.split("/")[-1]
        vsrc = (ginfo["vars"] if (choice.startswith("pressure_level/") and ginfo and ginfo.get("ok")) else minfo["vars"])
        vinfo = {v["name"]: v for v in vsrc}
        bname = base_name if base_name in vinfo else (
            "wind_u_" + base_name.split("wind_speed_")[-1] if base_name.startswith("wind_speed_") else base_name)
        vi = vinfo.get(bname)
        if vi and vi["extra"]:
            d0 = vi["extra"][0]
            vals = vi["extra_vals"].get(d0, [])
            if vals and all(isinstance(x, float) for x in vals):
                hp = [round(x / 100.0, 1) if max(vals) > 2000 else x for x in vals]
                kw["level"] = st.selectbox(f"Nível ({d0}, hPa)", hp,
                                           index=hp.index(850.0) if 850.0 in hp else 0, key=f"lvl_{mid}_{bname}")

        r1, r2 = st.columns([1, 3])
        r1.selectbox("Região", list(REGIONS), key="map_region", on_change=_on_region)
        with r2:
            cc = st.columns(5)
            cc[4].selectbox("Resolução", list(QUALITY), key="map_quality",
                            help="Limita as células por passo (memória). 'Alta' = grade nativa: em mapa global envia ~10 MB ao navegador e demora mais.")
            cc[0].number_input("Sul", -90.0, 90.0, key="bb_s", step=1.0)
            cc[1].number_input("Norte", -90.0, 90.0, key="bb_n", step=1.0)
            cc[2].number_input("Oeste", -180.0, 180.0, key="bb_w", step=1.0)
            cc[3].number_input("Leste", -180.0, 180.0, key="bb_e", step=1.0)
        bbox = (min(ss.bb_s, ss.bb_n), max(ss.bb_s, ss.bb_n), min(ss.bb_w, ss.bb_e), max(ss.bb_w, ss.bb_e))
        kw["max_cells"] = QUALITY[ss.map_quality]
        with st.expander("🎨 Aparência do mapa", expanded=False):
            ap = st.columns(4)
            ap[0].selectbox("Tema", list(MV.THEMES), key="map_theme")
            ap[1].selectbox("Estilo do campo", ["Suave", "Pixels", "Contornos"], key="map_style")
            ap[2].slider("Opacidade do campo", 0.3, 1.0, step=0.02, key="map_opacity")
            ap[3].checkbox("Escala robusta (P1–P99)", key="map_robust",
                           help="Ignora valores extremos para realçar o padrão espacial.")
            ap2 = st.columns(3)
            ap2[0].checkbox("Contorno dos países", key="map_borders")
            ap2[1].checkbox("Nomes dos países", key="map_labels", help="Aparecem ao dar zoom/escolher uma região.")
            ap2[2].checkbox("Terra de fundo", key="map_land")
        if st.button("🗺️ Gerar mapa", type="primary"):
            run_map(mid, choice, bbox, kw)

    mr = ss.get("map_res")
    if mr:
        st.divider()
        if "error" in mr:
            st.error(mr["error"])
        else:
            d = mr["data"]
            red = "" if d["members"] == 1 else f" · {d['reduction']} de {d['members']} membros"
            sub = (f"init {d['init']:%d/%m/%Y %HZ} · " if d["init"] is not None else "") + \
                  f"válido {d['valid']:%d/%m/%Y %HZ}{red}"
            fig = MV.build_map_figure(
                d, title=f"{C.pretty(mr['mid'])} · {d['label']}", subtitle=sub,
                colorscale=MV.PALETTES.get(ss.map_palette), theme=ss.map_theme, style=ss.map_style,
                opacity=ss.map_opacity, borders=ss.map_borders, labels=ss.map_labels, land=ss.map_land,
                robust=ss.map_robust)
            try:
                st.plotly_chart(fig, theme=None, config=MV.PLOT_CONFIG, **stretch(st.plotly_chart))
            except TypeError:
                st.plotly_chart(fig, config=MV.PLOT_CONFIG, **stretch(st.plotly_chart))
            f = d["field"]
            note = ""
            if d["stride"] > 1:
                note += f" · grade reamostrada (a cada {d['stride']} pontos) para caber na memória"
            if d["lat"].ndim == 2 and "u" in d:
                note += " · em grades curvilíneas as setas seguem os eixos da grade"
            st.caption(f"min {np.nanmin(f):.3g} · média {np.nanmean(f):.3g} · máx {np.nanmax(f):.3g} {d['units']}{note}"
                       " · use o ícone de câmera do gráfico para salvar PNG em alta resolução")
            st.download_button("⬇️ Baixar mapa interativo (HTML)",
                               fig.to_html(include_plotlyjs="cdn", config=MV.PLOT_CONFIG).encode("utf-8"),
                               file_name="mapa_previsao.html", mime="text/html")


# --------------------------------------------------------------------------
# Catálogo e variáveis
# --------------------------------------------------------------------------
def inspect_all(ids):
    prog = st.progress(0.0, text="Abrindo datasets…")
    infos = {}
    with ThreadPoolExecutor(max_workers=ss.workers + 2) as ex:
        futs = {ex.submit(C.inspect_dataset, i): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            infos[futs[fut]] = fut.result()
            prog.progress(n / len(ids), text=f"{n}/{len(ids)} · {C.pretty(futs[fut])}")
    prog.empty()
    ss.inspect_all = infos


with tab_cat:
    st.subheader(f"{len(IDS)} datasets · {SRC}")
    base = pd.DataFrame([{
        "dataset": i, "provedor": C.provider_of(i)[0],
        "tipo": "análise" if C.is_analysis_id(i) else "previsão",
        "otimização": "virtual (mapas, baixa latência)" if C.is_virtual(i) else "time-optimized (séries)",
        "cobertura": C.coverage_of(i)[0]} for i in IDS])
    if st.button("🔎 Inspecionar TODOS os datasets (abre cada um; pode levar alguns minutos)"):
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
                "lead máx (h)": v.get("lead_max_h", ""), "nº variáveis": len(v["vars"]) if v.get("ok") else ""})
        full = pd.concat([base, pd.DataFrame(rows)], axis=1)
        st.dataframe(full, hide_index=True, **stretch(st.dataframe))
        st.download_button("⬇️ Baixar tabela de datasets (CSV)", full.to_csv(index=False).encode("utf-8"),
                           file_name="datasets_dynamical.csv", mime="text/csv")
        names = sorted({x["name"] for v in allinf.values() if v.get("ok") for x in v["vars"]})
        mat = pd.DataFrame("", index=names, columns=IDS)
        for i, v in allinf.items():
            if v.get("ok"):
                for x in v["vars"]:
                    mat.loc[x["name"], i] = "✓"
        with st.expander(f"Matriz variável × dataset ({len(names)} variáveis distintas)"):
            st.dataframe(mat, **stretch(st.dataframe))
            st.download_button("⬇️ Baixar matriz (CSV)", mat.to_csv().encode("utf-8"),
                               file_name="matriz_variaveis.csv", mime="text/csv")
    else:
        st.dataframe(base, hide_index=True, **stretch(st.dataframe))
    st.divider()
    one = st.selectbox("Ver variáveis de um dataset", IDS, format_func=C.pretty, key="cat_one")
    with st.spinner("Abrindo…"):
        oi = C.inspect_dataset(one)
    if oi.get("ok"):
        st.dataframe(pd.DataFrame([{"variável": v["name"], "descrição": v["long_name"], "unidade": v["units"],
                                    "dimensões": ", ".join(v["dims"])} for v in oi["vars"]]),
                     hide_index=True, **stretch(st.dataframe))
    else:
        st.error(oi.get("error"))


# --------------------------------------------------------------------------
# Sobre
# --------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Como funciona.** O app lista os datasets com `dynamical_catalog.list()` e lê as variáveis direto de cada
dataset aberto (Zarr/Icechunk, leitura lazy): nada é baixado até você clicar em *Buscar e plotar* ou *Gerar mapa*.

**Séries** usam o ponto de grade mais próximo. **Conjuntos** (GEFS, IFS ENS, AIFS ENS, RRFS ENS, REFS, WeatherNext…)
aparecem como média e faixa P10–P90, ou como membros. **Derivadas**: velocidade/direção do vento (de u/v) e
precipitação acumulada (da taxa).

**Datasets *virtual*** são otimizados para mapas e baixa latência; séries neles leem muitos pedaços e demoram mais.
Datasets *time-optimized* são os melhores para séries.

**Cobertura.** Alguns modelos são regionais (HRRR, RTMA, URMA, RRFS, REFS, MRMS: EUA; ICON-EU: Europa;
HRDPS: Canadá). Fora do domínio o app avisa e pula o modelo.

**Atribuição.** Dados processados por dynamical.org. NOAA GFS/GEFS e ECMWF AIFS/IFS ENS aparecem como CC BY 4.0
(ECMWF também sob seus termos de uso). Para os demais, consulte a página do dataset em
[dynamical.org/catalog](https://dynamical.org/catalog).

**Mapa.** Global por padrão e independente de local. Interativo (zoom, arrastar, passar o mouse para ver valores), com tema escuro/claro, campo suave, em pixels ou
em contornos, setas de vento e contorno de todos os países (resolução ajustada ao zoom). Grades curvilíneas
(HRRR, HRDPS…) são reprojetadas para lat/lon por vizinho mais próximo só para exibição.
Limites de países: [Natural Earth](https://www.naturalearthdata.com) (domínio público), embutidos no projeto.

**Não incluído:** *Global Airport Observations (ASOS)* é GeoParquet, não Zarr.
""")
