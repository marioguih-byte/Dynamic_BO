# -*- coding: utf-8 -*-
"""
Núcleo de dados do app (sem dependência de Streamlit).

- Descobre TODOS os datasets do catálogo dynamical.org via `dynamical_catalog.list()`
- Descobre TODAS as variáveis lendo cada dataset (nomes, unidades, dimensões)
- Extrai séries em pontos e campos 2D em regiões, para qualquer grade
  (lat/lon regular ou curvilínea), previsão (init_time x lead_time) ou análise (time),
  determinístico ou conjunto (ensemble_member).
"""
from __future__ import annotations

import math
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------------------------------
# Catálogo
# --------------------------------------------------------------------------
# Lista de reserva (usada só se o catálogo ao vivo não puder ser consultado).
KNOWN_IDS = [
    "noaa-gfs-analysis", "noaa-gfs-forecast", "noaa-gfs-analysis-virtual", "noaa-gfs-forecast-virtual",
    "noaa-gefs-forecast-35-day", "noaa-gefs-analysis", "noaa-gefs-analysis-0-25-degree-virtual",
    "noaa-gefs-forecast-10-day-0-25-degree-virtual", "noaa-gefs-forecast-16-day-0-5-degree-virtual",
    "noaa-gefs-forecast-35-day-0-5-degree-virtual",
    "noaa-hrrr-forecast-18-hour-virtual", "noaa-hrrr-forecast-48-hour", "noaa-hrrr-forecast-48-hour-virtual",
    "noaa-hrrr-analysis", "noaa-hrrr-analysis-virtual",
    "noaa-rtma-analysis-virtual", "noaa-rtma-ru-analysis-virtual", "noaa-urma-analysis-virtual",
    "noaa-rrfs-forecast-sub-hourly-virtual", "noaa-rrfs-forecast-18-hour-virtual",
    "noaa-rrfs-forecast-84-hour-virtual", "noaa-rrfs-ens-forecast-virtual",
    "noaa-refs-forecast-hourly-virtual", "noaa-refs-forecast-3-hourly-virtual",
    "noaa-mrms-conus-analysis-hourly",
    "ecmwf-aifs-single-forecast", "ecmwf-aifs-single-forecast-virtual", "ecmwf-aifs-ens-forecast",
    "ecmwf-ifs-ens-forecast-15-day-0-25-degree", "ecmwf-ifs-ens-forecast-46-day-daily-1-5-degree",
    "ecmwf-ifs-ens-forecast-46-day-6-hourly-1-5-degree",
    "google-weathernext2-forecast-historical-virtual", "google-weathernext2-forecast-operational-virtual",
    "google-weathernext3-forecast-15-day-0-1-degree-virtual",
    "google-weathernext3-forecast-15-day-0-05-degree-virtual",
    "google-weathernext3-forecast-48-hour-0-1-degree-virtual",
    "google-weathernext3-forecast-48-hour-0-05-degree-virtual",
    "dwd-icon-eu-forecast-5-day",
    "nasa-imerg-analysis-early", "nasa-imerg-analysis-late",
    "ucsb-chc-chirps-analysis-final", "ucsb-chc-chirps-analysis-preliminary",
    "eccc-hrdps-forecast",
]

PROVIDERS = [
    ("noaa-gfs", "NOAA GFS"), ("noaa-gefs", "NOAA GEFS"), ("noaa-hrrr", "NOAA HRRR"),
    ("noaa-rtma-ru", "NOAA RTMA-RU"), ("noaa-rtma", "NOAA RTMA"), ("noaa-urma", "NOAA URMA"),
    ("noaa-rrfs-ens", "NOAA RRFS ENS"), ("noaa-rrfs", "NOAA RRFS"), ("noaa-refs", "NOAA REFS"),
    ("noaa-mrms", "NOAA MRMS"), ("ecmwf-aifs-single", "ECMWF AIFS Single"),
    ("ecmwf-aifs-ens", "ECMWF AIFS ENS"), ("ecmwf-ifs-ens", "ECMWF IFS ENS"),
    ("google-weathernext2", "Google WeatherNext 2"), ("google-weathernext3", "Google WeatherNext 3"),
    ("dwd-icon-eu", "DWD ICON-EU"), ("nasa-imerg", "NASA IMERG"), ("ucsb-chc-chirps", "UCSB CHC CHIRPS"),
    ("eccc-hrdps", "ECCC HRDPS"),
]
_PROV_SORTED = sorted(PROVIDERS, key=lambda p: -len(p[0]))

# Cobertura aproximada (heurística só para avisos/atalhos; a checagem real é feita nos dados)
REGIONAL = {
    "noaa-hrrr": ("EUA (CONUS)", (21, 53, -135, -60)), "noaa-rtma": ("EUA (CONUS)", (21, 53, -135, -60)),
    "noaa-urma": ("EUA (CONUS)", (21, 53, -135, -60)), "noaa-rrfs": ("EUA (CONUS)", (21, 53, -135, -60)),
    "noaa-refs": ("EUA (CONUS)", (21, 53, -135, -60)), "noaa-mrms": ("EUA (CONUS)", (21, 53, -135, -60)),
    "dwd-icon-eu": ("Europa", (29, 71, -24, 63)), "eccc-hrdps": ("Canadá", (38, 85, -150, -40)),
    "ucsb-chc-chirps": ("Global, terra 60°S–60°N", (-60, 60, -180, 180)),
}

STD_DIMS = {"init_time", "lead_time", "time", "ensemble_member"}
WIND_FACTORS = {"m/s": 1.0, "km/h": 3.6, "nós": 1.943844}
MASKS = {  # valores de preenchimento conhecidos (ver "validation report" dos datasets)
    "geopotential_height_cloud_ceiling": ("gt", 19000.0),
    "percent_frozen_precipitation_surface": ("lt", -0.1),
}


class OutOfDomain(ValueError):
    pass


def provider_of(dataset_id):
    for prefix, label in _PROV_SORTED:
        if dataset_id.startswith(prefix):
            return label, prefix
    parts = dataset_id.split("-")
    return " ".join(parts[:2]).upper(), "-".join(parts[:2])


def pretty(dataset_id):
    label, prefix = provider_of(dataset_id)
    rest = dataset_id[len(prefix):].strip("-").replace("-", " ")
    return f"{label} · {rest}" if rest else label


def is_virtual(dataset_id):
    return dataset_id.endswith("-virtual")


def is_analysis_id(dataset_id):
    return "-analysis" in dataset_id


def coverage_of(dataset_id):
    for prefix, (label, box) in REGIONAL.items():
        if dataset_id.startswith(prefix):
            return label, box
    return "Global", (-90, 90, -180, 180)


def covers_point(dataset_id, lat, lon):
    _, (s, n, w, e) = coverage_of(dataset_id)
    return s <= lat <= n and w <= lon <= e


def list_dataset_ids():
    """(ids, origem). Usa o catálogo ao vivo; se falhar, a lista embutida."""
    try:
        import dynamical_catalog
        ids = list(dynamical_catalog.list())
        if ids:
            return ids, "ao vivo (STAC do dynamical.org)"
    except Exception:
        pass
    return sorted(KNOWN_IDS), "lista embutida (catálogo ao vivo indisponível)"


# --------------------------------------------------------------------------
# Abertura/caches
# --------------------------------------------------------------------------
_DS, _INFO, _LATLON = {}, {}, {}
_LOCK = threading.Lock()


def clear_caches():
    with _LOCK:
        _DS.clear(); _INFO.clear(); _LATLON.clear()


_LAT_UNITS = {"degrees_north", "degree_north", "degrees_n", "degree_n", "degreen", "degreesn"}
_LON_UNITS = {"degrees_east", "degree_east", "degrees_e", "degree_e", "degreee", "degreese"}
_MEMBER_ALIASES = ("sample", "member", "number", "realization", "ens", "ensemble")


def _find_latlon(ds):
    """Nomes das coordenadas de latitude/longitude, por nome usual ou por atributos CF."""
    names = list(ds.variables)
    lower = {n.lower(): n for n in names}
    lat = next((lower[k] for k in ("latitude", "lat", "latitudes", "nav_lat", "y_lat", "lat_0") if k in lower), None)
    lon = next((lower[k] for k in ("longitude", "lon", "longitudes", "nav_lon", "x_lon", "lon_0") if k in lower), None)
    for n in names:
        a = ds[n].attrs
        sn, un = str(a.get("standard_name", "")).lower(), str(a.get("units", "")).lower().replace(" ", "")
        if lat is None and (sn == "latitude" or un in _LAT_UNITS):
            lat = n
        if lon is None and (sn == "longitude" or un in _LON_UNITS):
            lon = n
    return lat, lon


def _normalize(ds):
    """Padroniza nomes: latitude/longitude, ensemble_member e lead_time em timedelta."""
    la, lo = _find_latlon(ds)
    ren = {}
    if la and la != "latitude" and "latitude" not in ds.variables:
        ren[la] = "latitude"
    if lo and lo != "longitude" and "longitude" not in ds.variables:
        ren[lo] = "longitude"
    if "ensemble_member" not in ds.dims:
        alias = next((a for a in _MEMBER_ALIASES if a in ds.dims), None)
        if alias:
            ren[alias] = "ensemble_member"
    if ren:
        ds = ds.rename(ren)
    if "lead_time" in ds.coords and not np.issubdtype(ds["lead_time"].dtype, np.timedelta64):
        ds = ds.assign_coords(lead_time=pd.to_timedelta(ds["lead_time"].values, unit="s"))
    return ds


def get_ds(dataset_id, group=None):
    """Abre (lazy) um dataset do catálogo; `group` p.ex. 'pressure_level'."""
    key = (dataset_id, group)
    with _LOCK:
        ds = _DS.get(key)
    if ds is not None:
        return ds
    try:
        import dynamical_catalog
    except ImportError as exc:
        raise RuntimeError("Pacote ausente: pip install dynamical-catalog icechunk zarr") from exc
    kw = {"chunks": None}
    if group:
        kw["group"] = group
    ds = _normalize(dynamical_catalog.open(dataset_id, **kw))
    with _LOCK:
        _DS[key] = ds
    return ds


_GROUPS = {}


def list_groups(dataset_id):
    """Subgrupos Zarr do dataset (ex.: níveis de pressão), lidos do próprio store."""
    with _LOCK:
        if dataset_id in _GROUPS:
            return _GROUPS[dataset_id]
    try:
        import dynamical_catalog
        import zarr
        grp = zarr.open_group(dynamical_catalog.get_store(dataset_id), mode="r")
        names = sorted(str(k) for k in grp.group_keys())
    except Exception:
        names = ["pressure_level"]  # melhor palpite se a listagem não for possível
    with _LOCK:
        _GROUPS[dataset_id] = names
    return names


def list_groups_many(ids, workers=6):
    ids = list(ids)
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
        return dict(zip(ids, ex.map(list_groups, ids)))


def inspect_groups(ids, workers=6):
    """{grupo: {dataset_id: info}} para todos os subgrupos abríveis dos datasets."""
    gm = list_groups_many(ids, workers)
    pairs = [(i, g) for i, gs in gm.items() for g in gs]
    if not pairs:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as ex:
        res = list(ex.map(lambda p: inspect_dataset(p[0], p[1]), pairs))
    out = {}
    for (i, g), v in zip(pairs, res):
        if v.get("ok"):
            out.setdefault(g, {})[i] = v
    return out


def structure_report(dataset_id, group=None, max_vars=80):
    """Texto com dims, coordenadas, variáveis e grupos de um dataset (para diagnóstico)."""
    lines = [f"dataset: {dataset_id}" + (f"  ·  grupo: {group}" if group else "")]
    ds = None
    try:
        import dynamical_catalog
        kw = {"chunks": None}
        if group:
            kw["group"] = group
        ds = dynamical_catalog.open(dataset_id, **kw)
    except Exception as exc:  # noqa
        lines.append(f"ERRO ao abrir: {type(exc).__name__}: {exc}")
    if ds is not None:
        lines.append(f"dims: {dict(ds.sizes)}")
        lines.append("coordenadas:")
        for n, v in ds.coords.items():
            extra = ""
            if v.ndim == 1 and v.size:
                try:
                    extra = f"  [{v.values[0]} … {v.values[-1]}]"
                except Exception:  # noqa
                    pass
            lines.append(f"  - {n}  dims={tuple(v.dims)}  dtype={v.dtype}{extra}")
        lines.append(f"variáveis ({len(ds.data_vars)}):")
        for n, v in list(ds.data_vars.items())[:max_vars]:
            lines.append(f"  - {n}  dims={tuple(v.dims)}  units={v.attrs.get('units', '')}")
        if len(ds.data_vars) > max_vars:
            lines.append(f"  … (+{len(ds.data_vars) - max_vars})")
        la, lo = _find_latlon(ds)
        lines.append(f"lat/lon reconhecidos: {la!r} / {lo!r}")
    try:
        import dynamical_catalog
        import zarr
        g = zarr.open_group(dynamical_catalog.get_store(dataset_id), mode="r")
        if group:
            g = g[group]
        lines.append(f"subgrupos no store: {sorted(str(k) for k in g.group_keys())}")
        lines.append(f"arrays no nível: {sorted(str(k) for k in g.array_keys())[:60]}")
    except Exception as exc:  # noqa
        lines.append(f"(não foi possível listar o store: {type(exc).__name__})")
    return "\n".join(lines)


def lead_to_td(lead):
    return pd.to_timedelta(np.asarray(lead.values))


def norm_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


# --------------------------------------------------------------------------
# Estrutura do dataset
# --------------------------------------------------------------------------
def coord_names(ds):
    for a, b in (("latitude", "longitude"), ("lat", "lon")):
        if a in ds.variables and b in ds.variables:
            return a, b
    raise KeyError("dataset sem coordenadas latitude/longitude reconhecíveis "
                   f"(dims={dict(ds.sizes)}; coords={list(ds.coords)[:12]}; "
                   f"variáveis={list(ds.data_vars)[:6]}). Use 'Diagnosticar estrutura' para ver o dataset.")


def spatial_dims(ds):
    la, lo = coord_names(ds)
    dims = []
    for d in tuple(ds[la].dims) + tuple(ds[lo].dims):
        if d not in dims:
            dims.append(d)
    return tuple(dims)


def is_regular(ds):
    la, lo = coord_names(ds)
    return ds[la].ndim == 1 and ds[lo].ndim == 1


def _latlon2d(ds):
    key = id(ds)
    with _LOCK:
        v = _LATLON.get(key)
    if v is None:
        la, lo = coord_names(ds)
        v = (np.asarray(ds[la].values, float), np.asarray(ds[lo].values, float))
        with _LOCK:
            _LATLON[key] = v
    return v


def var_infos(ds):
    sd = set(spatial_dims(ds))
    out = []
    for name, da in ds.data_vars.items():
        if not sd.issubset(da.dims):
            continue
        extra = [d for d in da.dims if d not in sd and d not in STD_DIMS]
        extra_vals = {}
        for d in extra:
            if d in ds.coords:
                vals = np.asarray(ds[d].values)
                extra_vals[d] = [float(x) if np.issubdtype(vals.dtype, np.number) else str(x) for x in vals[:80]]
        out.append({"name": name,
                    "long_name": str(da.attrs.get("long_name") or da.attrs.get("standard_name") or name),
                    "units": str(da.attrs.get("units", "")), "dims": list(da.dims),
                    "extra": extra, "extra_vals": extra_vals})
    return out


def describe(ds):
    kind = ("forecast" if ("init_time" in ds.dims and "lead_time" in ds.dims)
            else "analysis" if "time" in ds.dims else "other")
    la, lo = coord_names(ds)
    reg = is_regular(ds)
    a, b = np.asarray(ds[la].values, float), np.asarray(ds[lo].values, float)
    d = {"kind": kind, "ens": "ensemble_member" in ds.dims, "members": int(ds.sizes.get("ensemble_member", 1)),
         "regular": reg, "sdims": list(spatial_dims(ds)),
         "lat": (float(np.nanmin(a)), float(np.nanmax(a))), "lon": (float(np.nanmin(b)), float(np.nanmax(b)))}
    if kind == "forecast":
        inits = ds["init_time"].values
        d["t0"], d["t1"] = str(pd.Timestamp(inits[0])), str(pd.Timestamp(inits[-1]))
        leads = np.asarray(lead_to_td(ds["lead_time"]).total_seconds()) / 3600.0
        d["leads_h"] = [float(x) for x in leads]
        d["lead_max_h"] = float(leads.max())
    elif kind == "analysis":
        times = ds["time"].values
        d["t0"], d["t1"] = str(pd.Timestamp(times[0])), str(pd.Timestamp(times[-1]))
    return d


def inspect_dataset(dataset_id, group=None):
    key = (dataset_id, group)
    with _LOCK:
        if key in _INFO:
            return _INFO[key]
    try:
        ds = get_ds(dataset_id, group)
        info = describe(ds)
        info.update(ok=True, vars=var_infos(ds))
    except Exception as exc:  # noqa
        fail = {"ok": False, "id": dataset_id, "group": group, "error": f"{type(exc).__name__}: {exc}"}
        if group is not None:  # grupo inexistente é esperado: não repetir a tentativa
            with _LOCK:
                _INFO[key] = fail
        return fail
    info.update(id=dataset_id, group=group)
    with _LOCK:
        _INFO[key] = info
    return info


def inspect_many(ids, group=None, workers=6):
    ids = list(ids)
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
        res = list(ex.map(lambda i: inspect_dataset(i, group), ids))
    return dict(zip(ids, res))


# --------------------------------------------------------------------------
# Variáveis derivadas
# --------------------------------------------------------------------------
_RATE_FACTORS = {  # unidade -> (mm acumulados) por segundo de intervalo
    "kg m-2 s-1": 1.0, "kg m**-2 s**-1": 1.0, "kg m^-2 s^-1": 1.0, "kg/m2/s": 1.0, "kg/m^2/s": 1.0,
    "kg s-1 m-2": 1.0, "mm/s": 1.0, "mm s-1": 1.0,
    "mm/h": 1 / 3600, "mm/hr": 1 / 3600, "mm h-1": 1 / 3600, "mm hr-1": 1 / 3600, "mm/hour": 1 / 3600,
    "mm/day": 1 / 86400, "mm d-1": 1 / 86400, "mm/d": 1 / 86400,
}


def _norm_units(units):
    return re.sub(r"\s+", " ", (units or "").lower().strip().replace("**", "^"))


def rate_factor(units):
    return _RATE_FACTORS.get(_norm_units(units))


_PRECIP_TOKENS = ("precip", "rain", "snowfall", "snow_water", "runoff")
_AMOUNT_UNITS = {"m", "meter", "metre", "meters", "metres", "kg m^-2", "kg m-2", "kg/m2", "kg/m^2", "mm"}
_PRETTY_UNITS = {"m s-1": "m/s", "m s^-1": "m/s", "w m-2": "W/m²", "w m^-2": "W/m²", "j m-2": "J/m²", "j m^-2": "J/m²",
                 "kg m-2": "kg/m²", "kg m^-2": "kg/m²", "1": "", "degree_celsius": "°C", "percent": "%"}


def _is_precip(name):
    n = name.lower()
    return any(t in n for t in _PRECIP_TOKENS) or bool(re.search(r"(?:^|_)tp(?:_|$)", n))  # ex.: imerg_tp_1hr


def unit_transform(name, mode, units, wind_unit="m/s"):
    """Conversão para unidades usadas no Brasil: valor_exibido = valor*scale + offset. -> (scale, offset, rótulo)."""
    n, u = name.lower(), _norm_units(units)
    if mode == "accum":
        return 1.0, 0.0, "mm"
    if mode == "wind_speed":
        return WIND_FACTORS[wind_unit], 0.0, wind_unit
    if mode == "wind_dir":
        return 1.0, 0.0, "°"
    f = rate_factor(units)
    if f is not None and _is_precip(n):                      # taxa -> mm/h
        return f * 3600.0, 0.0, "mm/h"
    if _is_precip(n) and u in _AMOUNT_UNITS:                   # lâmina: m -> mm; kg/m² = mm
        return (1000.0 if u in ("m", "meter", "metre", "meters", "metres") else 1.0), 0.0, "mm"
    if u in ("k", "kelvin"):
        return 1.0, -273.15, "°C"
    if u in ("degree_celsius", "degrees_celsius", "celsius", "degc", "°c"):
        return 1.0, 0.0, "°C"
    if u == "pa":
        return 0.01, 0.0, "hPa"
    if u in ("hpa", "mb", "mbar", "millibar", "millibars"):
        return 1.0, 0.0, "hPa"
    if u in ("m s^-1", "m s-1", "m/s") and n.startswith("wind_") and "direction" not in n:
        return WIND_FACTORS[wind_unit], 0.0, wind_unit
    if ("cloud_cover" in n or "cloud_fraction" in n or "relative_humidity" in n) and \
            u in ("1", "", "(0 - 1)", "(0-1)", "fraction", "0-1"):
        return 100.0, 0.0, "%"
    if u in ("percent", "%", "per cent"):
        return 1.0, 0.0, "%"
    if u in ("w m^-2", "w m-2", "w/m2", "w/m^2"):
        return 1.0, 0.0, "W/m²"
    if u in ("j m^-2", "j m-2", "j/m2", "j/m^2"):               # energia acumulada -> fluxo médio
        m = re.search(r"_(\d+)hr$", n)
        return (1.0 / (int(m.group(1)) * 3600.0), 0.0, "W/m²") if m else (1e-6, 0.0, "MJ/m²")
    if u in ("m^2 s^-2", "m2 s-2"):                            # geopotencial -> altura geopotencial
        return 1.0 / 9.80665, 0.0, "m"
    if "visib" in n and u == "m":
        return 0.001, 0.0, "km"
    return 1.0, 0.0, _PRETTY_UNITS.get(u, units or "")


def unit_label(key, units, wind_unit="m/s"):
    """Rótulo de unidade exibido para uma chave de variável (original ou derivada)."""
    k = key.split("/")[-1]
    if k.endswith("__acum"):
        return "mm"
    if k.startswith("wind_speed_"):
        return wind_unit
    if k.startswith("wind_direction_"):
        return "°"
    return unit_transform(k, "raw", units, wind_unit)[2]


def _is_amount_var(name):
    return bool(re.search(r"_(\d+)hr$", name))


def derived_defs(names, units_by_name, kind="forecast"):
    """Variáveis calculadas: velocidade/direção do vento e precipitação acumulada."""
    out = {}
    for n in sorted(names):
        m = re.match(r"^wind_u_(.+)$", n)
        if m and f"wind_v_{m.group(1)}" in names:
            h, bases = m.group(1), [n, f"wind_v_{m.group(1)}"]
            if f"wind_speed_{h}" not in names:
                out[f"wind_speed_{h}"] = dict(mode="wind_speed", bases=bases, units="m s-1",
                                              label=f"Velocidade do vento ({h}) · derivada de u/v")
            if f"wind_direction_{h}" not in names:
                out[f"wind_direction_{h}"] = dict(mode="wind_dir", bases=bases, units="°",
                                                  label=f"Direção do vento, de onde sopra ({h}) · derivada")
        if _is_precip(n) and rate_factor(units_by_name.get(n)) and kind in ("forecast", "analysis"):
            out[f"{n}__acum"] = dict(mode="accum", bases=[n], units="mm", label=f"{n} acumulada (mm) · derivada")
        elif (_is_precip(n) and rate_factor(units_by_name.get(n)) is None and _is_amount_var(n)
              and _norm_units(units_by_name.get(n)) in _AMOUNT_UNITS and kind in ("forecast", "analysis")):
            out[f"{n}__acum"] = dict(mode="accum_amt", bases=[n], units=units_by_name.get(n) or "m",
                                     label=f"{n} acumulada (mm) · derivada")
    return out


def union_vars(infos, group=None):
    """Une variáveis de vários datasets -> lista ordenada de dicts."""
    raw, der = {}, {}
    for info in infos.values():
        if not info.get("ok"):
            continue
        names = {v["name"] for v in info["vars"]}
        units = {v["name"]: v["units"] for v in info["vars"]}
        for v in info["vars"]:
            key = f"{group}/{v['name']}" if group else v["name"]
            e = raw.setdefault(key, {"key": key, "label": v["long_name"], "units": v["units"], "count": 0,
                                     "derived": False})
            e["count"] += 1
        if not group:
            for k, d in derived_defs(names, units, info["kind"]).items():
                e = der.setdefault(k, {"key": k, "label": d["label"], "units": d["units"], "count": 0,
                                       "derived": True})
                e["count"] += 1
    return [raw[k] for k in sorted(raw)] + [der[k] for k in sorted(der)]


# --------------------------------------------------------------------------
# Localização de pontos e seleção de tempo
# --------------------------------------------------------------------------
def point_dataset(ds, names, lat, lon):
    """Dataset (lazy) no ponto de grade mais próximo e (lat, lon) reais da grade."""
    la, lo = coord_names(ds)
    if is_regular(ds):
        lats, lons = np.asarray(ds[la].values, float), np.asarray(ds[lo].values, float)
        qlon = lon + 360.0 if (lons.max() > 180 and lon < 0) else lon
        tol_y = abs(np.diff(lats)).mean() if len(lats) > 1 else 1.0
        tol_x = abs(np.diff(lons)).mean() if len(lons) > 1 else 1.0
        if not (lats.min() - tol_y <= lat <= lats.max() + tol_y and lons.min() - tol_x <= qlon <= lons.max() + tol_x):
            raise OutOfDomain("ponto fora do domínio do dataset")
        sub = ds[names].sel({la: lat, lo: qlon}, method="nearest")
        return sub, (float(sub[la]), float(norm_lon(float(sub[lo]))))
    lat2, lon2 = _latlon2d(ds)
    dlon = norm_lon(lon2 - lon)
    d2 = (lat2 - lat) ** 2 + (dlon * math.cos(math.radians(lat))) ** 2
    iy, ix = np.unravel_index(np.nanargmin(d2), d2.shape)
    if math.sqrt(d2[iy, ix]) > 1.0:
        raise OutOfDomain("ponto fora do domínio do dataset")
    ydim, xdim = ds[la].dims
    sub = ds[names].isel({ydim: iy, xdim: ix})
    return sub, (float(lat2[iy, ix]), float(norm_lon(lon2[iy, ix])))


def _probe(da, **sel):
    """Valores de teste (primeiro membro / primeiro nível)."""
    da = da.sel(**sel) if sel else da
    for d in list(da.dims):
        if d == "ensemble_member" or (d not in STD_DIMS):
            da = da.isel({d: 0})
    return np.asarray(da.values)


def pick_init(da, requested=None, tries=6):
    inits = da["init_time"].values
    if requested is not None:
        cand = inits[inits <= np.datetime64(requested)]
        if len(cand) == 0:
            raise ValueError("init solicitado é anterior ao início do dataset")
        return cand[-1]
    for t in inits[::-1][:tries]:
        v = _probe(da.sel(init_time=t).isel(lead_time=slice(0, 4)))
        if np.isfinite(v).any():
            return t
    return inits[-1]


def pick_last_time(da, tries=6):
    times = da["time"].values
    for t in times[::-1][:tries]:
        if np.isfinite(_probe(da.sel(time=t))).any():
            return t
    return times[-1]


def _step_seconds(td, first_zero):
    s = np.asarray(td.total_seconds(), float)
    d = np.r_[np.nan, np.diff(s)]
    if len(d) > 1 and not first_zero:
        d[0] = d[1]
    return d


def _apply_mask(name, da):
    rule = MASKS.get(name)
    if not rule:
        return da
    return da.where(da <= rule[1]) if rule[0] == "gt" else da.where(da >= rule[1])


def _is_level_dim(d):
    return any(t in d.lower() for t in ("pressure", "level", "plev", "isobaric"))


def _load_parallel(sub, workers):
    names = list(sub.data_vars)
    if workers <= 1 or len(names) <= 1:
        return sub.load()
    with ThreadPoolExecutor(max_workers=min(workers, len(names))) as ex:
        arrs = list(ex.map(lambda n: sub[n].load(), names))
    return xr.Dataset({n: a for n, a in zip(names, arrs)})


def _members_first(da, tdim):
    if "ensemble_member" in da.dims:
        da = da.transpose("ensemble_member", tdim)
    return np.asarray(da.values, dtype="float64")


def parse_choice(choice):
    if "/" in choice:
        g, k = choice.split("/", 1)
        return g, k
    return None, choice


def _compute(mode, bases, sub):
    if mode == "raw":
        return _apply_mask(bases[0], sub[bases[0]])
    if mode == "wind_speed":
        return np.hypot(sub[bases[0]], sub[bases[1]])
    if mode == "wind_dir":
        return (270.0 - np.degrees(np.arctan2(sub[bases[1]], sub[bases[0]]))) % 360.0
    raise ValueError(mode)


# --------------------------------------------------------------------------
# Séries temporais em um ponto
# --------------------------------------------------------------------------
def fetch_series(dataset_id, lat, lon, choices, *, init_req=None, horizon_days=7.0, analysis_days=3.0,
                 levels=(850, 500), var_workers=4):
    """
    choices: lista de chaves ('temperature_2m', 'wind_speed_10m', 'precipitation_surface__acum',
             'pressure_level/temperature', ...). Variáveis ausentes no dataset são puladas.
    """
    by_group = {}
    for ch in choices:
        g, k = parse_choice(ch)
        by_group.setdefault(g, []).append(k)
    notes, skipped, usable = [], [], {}
    for g, keys in by_group.items():  # só usa grupos/raiz que abrem e têm grade
        try:
            ds_g = get_ds(dataset_id, g)
            usable[g] = describe(ds_g)
        except Exception as exc:  # noqa
            skipped += [f"{g}/{k}" if g else k for k in keys]
            notes.append(f"{g or 'raiz'}: {type(exc).__name__}: {str(exc)[:110]}")
    if not usable:
        raise KeyError("; ".join(notes) or "nenhuma variável utilizável")
    kind = next(iter(usable.values()))["kind"]
    if kind == "other":
        raise ValueError("dataset sem dimensão temporal reconhecida (init_time/lead_time ou time)")
    tdim = "lead_time" if kind == "forecast" else "time"

    out = {"id": dataset_id, "kind": kind, "init": None, "valid": None, "grid": None,
           "ens": False, "vars": {}, "skipped": skipped, "notes": notes}
    init_t = last_t = None
    for g, keys in by_group.items():
        if g not in usable:
            continue
        ds = get_ds(dataset_id, g)
        infos = {v["name"]: v for v in var_infos(ds)}
        units = {n: v["units"] for n, v in infos.items()}
        ddefs = derived_defs(set(infos), units, kind)
        specs = []
        for k in keys:
            label_k = f"{g}/{k}" if g else k
            if k in infos:
                specs.append((label_k, "raw", [k], infos[k]["units"], infos[k]["long_name"], None))
            elif k in ddefs:
                d = ddefs[k]
                fct = rate_factor(units.get(d["bases"][0])) if d["mode"] == "accum" else None
                specs.append((label_k, d["mode"], d["bases"], d["units"], d["label"], fct))
            else:
                out["skipped"].append(label_k)
        if not specs:
            continue
        need = sorted({b for s in specs for b in s[2]})
        sub, grid = point_dataset(ds, need, lat, lon)
        out["grid"] = out["grid"] or grid
        probe_name = next((p for p in ("temperature_2m", "wind_u_10m", "pressure_surface") if p in need), need[0])
        if kind == "forecast":
            if init_t is None:
                init_t = pick_init(sub[probe_name], init_req)
            sub = sub.sel(init_time=init_t, method="nearest")
            td_all = lead_to_td(sub["lead_time"])
            keep = max(1, int((td_all <= pd.Timedelta(days=horizon_days)).sum()))
            sub = sub.isel(lead_time=slice(0, keep))
        else:
            if last_t is None:
                last_t = pick_last_time(sub[probe_name])
            start = last_t - np.timedelta64(int(analysis_days * 24), "h")
            sub = sub.sel(time=slice(start, last_t))
        sub = _load_parallel(sub, var_workers)
        if kind == "forecast":
            td = lead_to_td(sub["lead_time"])
            valid = pd.Timestamp(init_t) + td
            dt = _step_seconds(td, first_zero=True)
            out["init"] = pd.Timestamp(init_t)
        else:
            valid = pd.DatetimeIndex(sub["time"].values)
            dt = _step_seconds(pd.to_timedelta(np.asarray(sub["time"].values - sub["time"].values[0])),
                               first_zero=False)
        out["valid"] = out["valid"] if out["valid"] is not None else valid
        for key, mode, bases, unit, label, fct in specs:
            if mode == "accum":
                pr = sub[bases[0]]
                dtx = xr.DataArray(dt, dims=tdim)
                da = (pr * fct * dtx).fillna(0).cumsum(tdim)
                da = da.where(pr.notnull() | (dtx.isnull() & (kind == "forecast")))
            elif mode == "accum_amt":  # soma das lâminas por intervalo (ex.: 1 h)
                pr = sub[bases[0]]
                da = pr.fillna(0).cumsum(tdim).where(pr.notnull())
                w = re.search(r"_(\d+)hr$", bases[0])
                med = float(np.nanmedian(dt[1:])) if len(dt) > 1 else float("nan")
                if w and np.isfinite(med) and abs(med - int(w.group(1)) * 3600.0) > 0.02 * int(w.group(1)) * 3600.0:
                    out["notes"].append(f"{key}: passo de {med / 3600:g} h difere da janela de {w.group(1)} h; "
                                        "o acumulado é aproximado")
            else:
                da = _compute(mode, bases, sub)
            extras = [d for d in da.dims if d not in (tdim, "ensemble_member")]
            items = [("", da)]
            if extras:
                d0 = extras[0]
                da = da.isel({d: 0 for d in extras[1:]}) if len(extras) > 1 else da
                if d0 in da.coords and _is_level_dim(d0):
                    vals = np.asarray(da[d0].values, float)
                    items = []
                    for L in levels:
                        Lc = L * 100.0 if np.nanmax(vals) > 2000 else float(L)
                        items.append((f"@{L} hPa", da.sel({d0: Lc}, method="nearest")))
                else:
                    items = [(f"@{d0}[0]", da.isel({d0: 0}))]
            for suffix, item in items:
                arr = _members_first(item, tdim)
                out["ens"] = out["ens"] or arr.ndim == 2
                if mode in ("accum", "accum_amt") and kind == "forecast":
                    arr[..., 0] = 0.0
                if not np.isfinite(arr).any():
                    out.setdefault("notes", []).append(f"{key + suffix}: sem dados nesta rodada/período")
                out["vars"][key + suffix] = {"name": key.split("/")[-1], "mode": mode, "units": unit,
                                             "label": label + (f" {suffix}" if suffix else ""),
                                             "data": arr, "valid": valid}
    if not out["vars"]:
        raise KeyError("nenhuma das variáveis escolhidas existe neste dataset")
    return out


# --------------------------------------------------------------------------
# Conversão/estatística para exibição
# --------------------------------------------------------------------------
def display_values(meta, arr, wind_unit="m/s"):
    scale, off, label = unit_transform(meta.get("name", ""), meta.get("mode", "raw"), meta.get("units", ""), wind_unit)
    return arr * scale + off, label


def summarize(arr):
    if arr.ndim == 1:
        return arr, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, 0), np.nanpercentile(arr, 10, 0), np.nanpercentile(arr, 90, 0)


def results_to_dataframe(results, tz_h=0.0, wind_unit="m/s"):
    frames = []
    for did, res in results.items():
        for key, meta in res["vars"].items():
            valid = meta.get("valid", res["valid"])
            arr, unit = display_values(meta, meta["data"], wind_unit)
            m, lo, hi = summarize(arr)
            df = pd.DataFrame({"dataset": did, "init_utc": res["init"], "valid_utc": valid,
                               "valid_local": valid + pd.Timedelta(hours=tz_h),
                               "variavel": key, "unidade": unit, "valor": m,
                               "p10": lo if lo is not None else np.nan,
                               "p90": hi if hi is not None else np.nan,
                               "lat_grade": res["grid"][0], "lon_grade": res["grid"][1]})
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Mapas
# --------------------------------------------------------------------------
def _region_slices(ds, bbox, max_cells):
    s, n, w, e = bbox
    la, lo = coord_names(ds)
    if is_regular(ds):
        lats, lons = np.asarray(ds[la].values, float), np.asarray(ds[lo].values, float)
        lons_n = norm_lon(lons) if lons.max() > 180 else lons
        ii = np.where((lats >= s) & (lats <= n))[0]
        jj = np.where((lons_n >= w) & (lons_n <= e))[0]
        if len(ii) == 0 or len(jj) == 0:
            raise OutOfDomain("região fora do domínio do dataset")
        sl = {la: slice(int(ii.min()), int(ii.max()) + 1), lo: slice(int(jj.min()), int(jj.max()) + 1)}
        ny, nx = len(ii), len(jj)
    else:
        lat2, lon2 = _latlon2d(ds)
        mask = (lat2 >= s) & (lat2 <= n) & (norm_lon(lon2) >= w) & (norm_lon(lon2) <= e)
        if not mask.any():
            raise OutOfDomain("região fora do domínio do dataset")
        rows, cols = np.where(mask)
        ydim, xdim = ds[la].dims
        sl = {ydim: slice(int(rows.min()), int(rows.max()) + 1), xdim: slice(int(cols.min()), int(cols.max()) + 1)}
        ny, nx = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
    k = max(1, math.ceil(math.sqrt(ny * nx / max_cells)))
    if k > 1:
        sl = {d: slice(x.start, x.stop, k) for d, x in sl.items()}
    return sl, k


def _select_extra(obj, level, spatial):
    for d in [d for d in obj.dims if d not in STD_DIMS and d not in spatial]:
        if d in obj.coords and _is_level_dim(d) and level is not None:
            vals = np.asarray(obj[d].values, float)
            L = level * 100.0 if np.nanmax(vals) > 2000 else float(level)
            obj = obj.sel({d: L}, method="nearest")
        else:
            obj = obj.isel({d: 0})
    return obj


def fetch_map(dataset_id, choice, bbox, *, init_req=None, lead_h=24.0, time_req=None,
              member_mode="mean", member_idx=0, level=None, max_cells=600_000, wind_unit="m/s"):
    """
    Campo 2D em `bbox=(S,N,W,E)`. Para variáveis acumuladas: soma de 0 a lead_h.
    member_mode: mean | std | max | min | member.
    """
    g, key = parse_choice(choice)
    ds = get_ds(dataset_id, g)
    info = describe(ds)
    kind = info["kind"]
    vi = {v["name"]: v for v in var_infos(ds)}
    units = {n: v["units"] for n, v in vi.items()}
    ddefs = derived_defs(set(vi), units, kind)
    if key in vi:
        mode, bases, unit, label, fct = "raw", [key], vi[key]["units"], vi[key]["long_name"], None
    elif key in ddefs and ddefs[key]["mode"] != "wind_dir":
        d = ddefs[key]
        mode, bases, unit, label = d["mode"], d["bases"], d["units"], d["label"]
        fct = rate_factor(units.get(bases[0])) if mode == "accum" else None
    else:
        raise KeyError(f"variável '{key}' indisponível para mapa neste dataset")
    acc = mode in ("accum", "accum_amt")
    if acc and kind != "forecast":
        raise ValueError("precipitação acumulada em mapa só está disponível para datasets de previsão")

    sdims = spatial_dims(ds)
    sl, stride = _region_slices(ds, bbox, max_cells)
    region = ds[bases].isel(sl)
    region = _select_extra(region, level, sdims)
    la, lo = coord_names(ds)

    center = region[bases[0]].isel({d: region.sizes[d] // 2 for d in sdims})
    if kind == "forecast":
        init_t = pick_init(center, init_req)
        region = region.sel(init_time=init_t, method="nearest")
        target = pd.Timedelta(hours=lead_h)
        if acc:
            region = region.sel(lead_time=slice(pd.Timedelta(0), target))
            td = lead_to_td(region["lead_time"])
            dt = _step_seconds(td, first_zero=True)
            valid = pd.Timestamp(init_t) + target
        else:
            region = region.sel(lead_time=target, method="nearest")
            valid = pd.Timestamp(init_t) + pd.Timedelta(region["lead_time"].values)
        tdim, steps, init_out = "lead_time", (region.sizes.get("lead_time", 1) if acc else 1), pd.Timestamp(init_t)
    elif kind == "analysis":
        t = np.datetime64(time_req) if time_req is not None else pick_last_time(center)
        region = region.sel(time=t, method="nearest")
        valid, tdim, steps, init_out = pd.Timestamp(region["time"].values), None, 1, None
    else:
        raise ValueError("dataset sem dimensão temporal reconhecida")

    ny = int(np.prod([region.sizes[d] for d in sdims]))
    n_mem = region.sizes.get("ensemble_member", 1)
    per_member = ny * max(steps, 1) * len(bases)
    if per_member > 4e7:
        raise MemoryError("região/horizonte grandes demais; reduza a resolução, a região ou o passo")
    block = max(1, int(2e7 // per_member))
    if member_mode == "member":
        idxs = [min(max(0, int(member_idx)), n_mem - 1)]
    else:
        idxs = list(range(n_mem))

    s = c = sq = u_s = v_s = mx = mn = None
    for i in range(0, len(idxs), block):
        sel = idxs[i:i + block]
        blk = region.isel(ensemble_member=sel) if "ensemble_member" in region.dims else region
        blk = blk.load()
        if mode == "accum_amt":
            f = blk[bases[0]].fillna(0).sum("lead_time")
        elif mode == "accum":
            dtx = xr.DataArray(dt, dims="lead_time")
            f = (blk[bases[0]] * fct * dtx).fillna(0).sum("lead_time")
        elif mode == "wind_speed":
            f = np.hypot(blk[bases[0]], blk[bases[1]])
        else:
            f = _apply_mask(bases[0], blk[bases[0]])
        if "ensemble_member" not in f.dims:
            f = f.expand_dims("ensemble_member")
        a = f.transpose("ensemble_member", *sdims).values.astype("float64")
        fin = np.isfinite(a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _s, _c = np.where(fin, a, 0).sum(0), fin.sum(0)
            _sq = np.where(fin, a * a, 0).sum(0)
            _mx, _mn = np.nanmax(a, 0) if a.size else a, np.nanmin(a, 0) if a.size else a
        s = _s if s is None else s + _s
        c = _c if c is None else c + _c
        sq = _sq if sq is None else sq + _sq
        mx = _mx if mx is None else np.fmax(mx, _mx)
        mn = _mn if mn is None else np.fmin(mn, _mn)
        if mode == "wind_speed":
            for nm, store in ((bases[0], "u"), (bases[1], "v")):
                x = blk[nm]
                if "ensemble_member" in x.dims:
                    x = x.mean("ensemble_member")
                x = x.transpose(*sdims).values.astype("float64") * len(sel)
                if store == "u":
                    u_s = x if u_s is None else u_s + x
                else:
                    v_s = x if v_s is None else v_s + x
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
        field = {"mean": mean, "std": np.sqrt(np.maximum(sq / np.maximum(c, 1) - mean ** 2, 0)),
                 "max": mx, "min": mn, "member": mean}[member_mode]
    scale, off, unit_out = unit_transform(key, mode, unit, wind_unit)   # unidades usadas no Brasil
    field = field * scale if member_mode == "std" else field * scale + off  # desvio-padrão não leva deslocamento
    res = {"init": init_out, "valid": valid, "field": field, "units": unit_out, "label": label, "stride": stride,
           "lat": np.asarray(region[la].values), "lon": np.asarray(region[lo].values), "key": choice,
           "reduction": member_mode if n_mem > 1 else "determinístico", "members": n_mem}
    if mode == "wind_speed":
        res["u"], res["v"] = u_s / len(idxs), v_s / len(idxs)
    if res["lon"].ndim == 1 and np.nanmax(res["lon"]) > 180:
        res["lon"] = norm_lon(res["lon"])
    return res
