#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nemo_and_histo_profiles.py (ALIGNED with Yamazaki DA) — FIXED METHOD
-------------------------------------------------------------------
Attendu (selon tes consignes) :

FIG.1 (profils):
- Points bleus : historique NEMO aux positions d'observations (CSV Datas_OK racine)
- Courbe noire + ±σ : NEMO récent 2005–2023 calculé sur TOUTE la grille (NetCDF),
  restreint à la même mer / latband, même saison.
- Binning vertical affichage : 1 m (profondeur entière), interpolation linéaire
  depuis niveaux NetCDF, APRES calcul des mu_native et sigma_native.

FIG.2 (scatter):
- Comparaison point-par-point (comme avant) :
    x = nemo_hist_T
    y = T_ref = moyenne (2005–2023) de nemo_recent_T pour ce point historique précis
- T_ref (et std) sont recalculés depuis les CSV via recent_year + nemo_recent_T.
  On n'utilise PAS un profil spatial NetCDF pour le scatter.

IMPORTANT:
- csv-dir = Datas_OK (racine)
- nemo-nc = thetao_1900-1969_2005-2023_60S-90S_10-200m_MONTHLY.nc
"""

import argparse
import glob
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator  # <-- AJOUT (ticks pas=1)

import xarray as xr


# =============================================================================
# PARAMÈTRES VISUELS (alignés sur Yamazaki, DA conservée)
# =============================================================================

LEGEND_FONTSIZE = 11
LEGEND_TITLE_FONTSIZE = 11
LEGEND_Y = 0.020

plt.rcParams.update({
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_TITLE_FONTSIZE,
})

SEASON_MONTHS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'ALL': list(range(1, 13)),
}

# =============================================================================
# DÉFINITION DES MERS (0-360° sens horaire) — identique à vos conventions
# =============================================================================

SEAS = {
    'Ross': {
        'lat_min': -78, 'lat_max': -70,
        'lon_start': 160,
        'lon_end': 210,
        'label': 'Ross'
    },
    'Weddell': {
        'lat_min': -78, 'lat_max': -60,
        'lon_start': 300,
        'lon_end': 340,
        'label': 'Weddell'
    },
    'Bellingshausen-Amundsen': {
        'lat_min': -75, 'lat_max': -65,
        'lon_start': 230,
        'lon_end': 300,
        'label': 'Bellingshausen/\nAmundsen'
    },
    'Davis': {
        'lat_min': -68, 'lat_max': -65,
        'lon_start': 80,
        'lon_end': 100,
        'label': 'Davis'
    }
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def normalize_longitude_360(lon):
    return lon % 360


def filter_by_sea(df, sea_config, lon_col='hist_lon', lat_col='hist_lat'):
    df = df.copy()
    df['lon_360'] = normalize_longitude_360(df[lon_col])

    df_sea = df[
        (df[lat_col] >= sea_config['lat_min']) &
        (df[lat_col] < sea_config['lat_max'])
    ]

    lon_start = sea_config['lon_start']
    lon_end = sea_config['lon_end']

    if lon_start <= lon_end:
        df_sea = df_sea[
            (df_sea['lon_360'] >= lon_start) & (df_sea['lon_360'] < lon_end)
        ]
    else:
        df_sea = df_sea[
            (df_sea['lon_360'] >= lon_start) |
            (df_sea['lon_360'] < lon_end)
        ]

    return df_sea


def _compute_xlim_from_content(x_values, pad_abs=0.25, pad_frac=0.05):
    vals = []
    for v in x_values:
        if v is None:
            continue
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)

    if not vals:
        return (-2.5, 4.5)

    allv = np.concatenate(vals)
    vmin = float(np.min(allv))
    vmax = float(np.max(allv))
    span = max(1e-6, vmax - vmin)
    pad = max(pad_abs, pad_frac * span)
    return (vmin - pad, vmax + pad)


# =============================================================================
# NetCDF NEMO helpers
# =============================================================================

def _guess_varname_thetao(ds: xr.Dataset) -> str:
    for cand in ["thetao", "toce", "temp", "temperature"]:
        if cand in ds.data_vars:
            return cand
    for v in ds.data_vars:
        if ds[v].dtype.kind in "fc" and ds[v].ndim >= 3:
            return v
    raise KeyError("Impossible de trouver la variable température (thetao) dans le NetCDF.")


def _guess_coord(ds: xr.Dataset, names):
    for n in names:
        if n in ds.coords:
            return n
        if n in ds.variables:
            return n
    return None


def _to_lon360(lon):
    return lon % 360


def _build_region_mask(lat2d, lon2d, lat_min, lat_max, lon_start, lon_end):
    lon360 = _to_lon360(lon2d)
    mlat = (lat2d >= lat_min) & (lat2d < lat_max)
    if lon_start <= lon_end:
        mlon = (lon360 >= lon_start) & (lon360 < lon_end)
    else:
        mlon = (lon360 >= lon_start) | (lon360 < lon_end)
    return mlat & mlon


def _build_latband_mask(lat2d, lon2d, lat_min_abs, lat_max_abs):
    # lat_min_abs=60, lat_max_abs=70 means [-70,-60)
    return (lat2d >= -lat_max_abs) & (lat2d < -lat_min_abs)


def _interp_profile_to_1m(depth_native, prof_native, depths_i):
    """
    Interpolation linéaire sur grille 1 m.
    IMPORTANT: à utiliser APRES calcul des stats (mu_native/sigma_native),
    et pas sur données brutes.
    """
    depth_native = np.asarray(depth_native, dtype=float)
    prof_native = np.asarray(prof_native, dtype=float)
    ok = np.isfinite(depth_native) & np.isfinite(prof_native)
    if ok.sum() < 2:
        return np.full_like(depths_i, np.nan, dtype=float)
    dn = depth_native[ok]
    pn = prof_native[ok]
    order = np.argsort(dn)
    dn = dn[order]
    pn = pn[order]
    out = np.interp(depths_i, dn, pn, left=np.nan, right=np.nan)
    return out


class NemoRecentProvider:
    """
    Fournit mu(z) et sigma(z) pour 2005–2023 depuis le NetCDF,
    restreint à une saison et une région (mer ou latband).
    Puis interpolation 1 m APRES calcul des stats.
    """
    def __init__(self, nc_path: str, season: str, depth_min: float, depth_max: float):
        self.nc_path = nc_path
        self.season = season
        self.months = SEASON_MONTHS[season]
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.ds = xr.open_dataset(nc_path)
        self.varT = _guess_varname_thetao(self.ds)

        self.time_name = _guess_coord(self.ds, ["time", "t", "time_counter"])
        self.depth_name = _guess_coord(self.ds, ["depth", "deptht", "lev", "z", "depthu", "depthv"])
        self.lat_name = _guess_coord(self.ds, ["nav_lat", "lat", "latitude", "y"])
        self.lon_name = _guess_coord(self.ds, ["nav_lon", "lon", "longitude", "x"])

        if self.time_name is None or self.depth_name is None:
            raise KeyError("NetCDF: coordonnées time/depth introuvables.")
        if self.lat_name is None or self.lon_name is None:
            raise KeyError("NetCDF: coordonnées lat/lon introuvables.")

        self.lat2d = self.ds[self.lat_name].values
        self.lon2d = self.ds[self.lon_name].values

        dmin_i = int(np.ceil(depth_min))
        dmax_i = int(np.floor(depth_max))
        self.depths_i = np.arange(dmin_i, dmax_i + 1, 1)

    @lru_cache(maxsize=64)
    def recent_profile_sea(self, sea_name: str):
        conf = SEAS[sea_name]
        mask = _build_region_mask(self.lat2d, self.lon2d,
                                  conf['lat_min'], conf['lat_max'],
                                  conf['lon_start'], conf['lon_end'])
        return self._compute_recent_profile_from_mask(mask)

    @lru_cache(maxsize=64)
    def recent_profile_latband(self, lat_min_abs: int, lat_max_abs: int, sea_name: str | None):
        mask = _build_latband_mask(self.lat2d, self.lon2d, lat_min_abs, lat_max_abs)
        if sea_name is not None:
            conf = SEAS[sea_name]
            mask_sea = _build_region_mask(self.lat2d, self.lon2d,
                                          conf['lat_min'], conf['lat_max'],
                                          conf['lon_start'], conf['lon_end'])
            mask = mask & mask_sea
        return self._compute_recent_profile_from_mask(mask)

    def _compute_recent_profile_from_mask(self, mask2d):
        daT = self.ds[self.varT]
        t = self.ds[self.time_name]

        years = t.dt.year
        months = t.dt.month
        sel_time = (years >= 2005) & (years <= 2023) & months.isin(self.months)

        da = daT.sel({self.time_name: sel_time})

        depth = self.ds[self.depth_name]
        da = da.sel({self.depth_name: depth.where((depth >= self.depth_min) & (depth <= self.depth_max), drop=True)})

        da_masked = da.where(mask2d)

        dims = list(da_masked.dims)
        red_dims = [d for d in dims if d not in (self.time_name, self.depth_name)]

        mu_native = da_masked.mean(dim=[self.time_name] + red_dims, skipna=True)
        s_native  = da_masked.std(dim=[self.time_name] + red_dims, skipna=True, ddof=1)

        depth_native = mu_native[self.depth_name].values
        mu_native_v = mu_native.values
        s_native_v  = s_native.values

        mu_1m = _interp_profile_to_1m(depth_native, mu_native_v, self.depths_i)
        s_1m  = _interp_profile_to_1m(depth_native, s_native_v,  self.depths_i)

        return self.depths_i, mu_1m, s_1m


# =============================================================================
# CHARGEMENT & AGRÉGATION CSV : points uniques + T_ref point-par-point
# =============================================================================

def _iter_csv_files(csv_dir: Path, season: str):
    pattern = str(csv_dir / f"nemo_yamazaki_{season}_*.csv")
    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


def _required_columns_present(header_cols: list[str], needed: list[str]) -> bool:
    s = set(header_cols)
    return all(c in s for c in needed)


def load_unique_points_with_tref(csv_dir, season, depth_min, depth_max, y0, y1,
                                 chunksize=1_000_000):
    """
    Construit un DataFrame "points uniques" (1 ligne par point historique),
    et calcule T_ref / std / n depuis (recent_year, nemo_recent_T) point-par-point.

    On ignore les colonnes recent_mean/recent_std si elles sont partiellement remplies :
    le calcul est refait proprement et exhaustivement.

    Sortie colonnes:
    - hist_year, hist_month, hist_day, hist_lat, hist_lon, hist_depth_m, nemo_hist_T
    - T_ref, T_ref_std, T_ref_n
    """
    csv_dir = Path(csv_dir).expanduser().resolve()
    files = _iter_csv_files(csv_dir, season)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé: {csv_dir}/nemo_yamazaki_{season}_*.csv")

    season_months = set(SEASON_MONTHS[season])

    # Clé stricte du point historique (exactement comme votre awk)
    key_cols = ["hist_year", "hist_month", "hist_day", "hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T"]

    # Accumulateurs global dict (clé string -> sums, sumsq, cnt)
    sum_d = {}
    sumsq_d = {}
    cnt_d = {}
    # On garde aussi un mapping clé -> tuple valeurs (pour reconstruire DF sans re-split)
    meta_d = {}

    usecols = key_cols + ["recent_year", "nemo_recent_T"]

    for f in files:
        # lecture en chunks
        try:
            it = pd.read_csv(f, usecols=lambda c: c in set(usecols), chunksize=chunksize)
        except ValueError:
            # si usecols lambda échoue (colonnes manquantes), on diagnostique
            hdr = pd.read_csv(f, nrows=0).columns.tolist()
            missing = [c for c in usecols if c not in hdr]
            raise ValueError(f"Colonnes manquantes dans {f.name}: {missing}")

        for chunk in it:
            # coercition numérique minimale
            for c in ["hist_year", "hist_month", "hist_day", "recent_year"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            for c in ["hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T", "nemo_recent_T"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            # filtres
            chunk = chunk[chunk["hist_month"].isin(season_months)]
            chunk = chunk[(chunk["hist_year"] >= y0) & (chunk["hist_year"] < y1)]
            chunk = chunk[(chunk["hist_depth_m"] >= max(10, depth_min)) & (chunk["hist_depth_m"] <= depth_max)]
            chunk = chunk[chunk["nemo_hist_T"].between(-10, 10)]
            chunk = chunk.dropna(subset=key_cols + ["recent_year", "nemo_recent_T"])

            if chunk.empty:
                continue

            # recent window 2005–2023
            chunk = chunk[(chunk["recent_year"] >= 2005) & (chunk["recent_year"] <= 2023)]
            if chunk.empty:
                continue

            # group within chunk
            g = chunk.groupby(key_cols, dropna=False)["nemo_recent_T"]
            sum_s = g.sum()
            cnt_s = g.count()
            sumsq_s = g.apply(lambda s: float(np.sum(np.square(s.to_numpy(dtype=float)))))

            # update dicts
            for k, s in sum_s.items():
                # k est tuple selon key_cols
                key_tuple = k if isinstance(k, tuple) else (k,)
                key_str = ",".join(map(str, key_tuple))

                c = int(cnt_s.loc[k])
                sq = float(sumsq_s.loc[k])

                sum_d[key_str] = sum_d.get(key_str, 0.0) + float(s)
                sumsq_d[key_str] = sumsq_d.get(key_str, 0.0) + sq
                cnt_d[key_str] = cnt_d.get(key_str, 0) + c

                if key_str not in meta_d:
                    meta_d[key_str] = key_tuple

    if not cnt_d:
        return pd.DataFrame()

    # Build DF
    rows = []
    for key_str, n in cnt_d.items():
        key_tuple = meta_d[key_str]
        s = sum_d[key_str]
        sq = sumsq_d[key_str]

        n = int(n)
        mu = s / n if n > 0 else np.nan

        if n >= 2:
            var = (sq - (s * s) / n) / (n - 1)
            var = max(0.0, float(var))
            sd = float(np.sqrt(var))
        else:
            sd = np.nan

        row = dict(zip(key_cols, key_tuple))
        row["T_ref"] = float(mu)
        row["T_ref_std"] = float(sd) if np.isfinite(sd) else np.nan
        row["T_ref_n"] = n
        rows.append(row)

    df_points = pd.DataFrame(rows)

    # dtypes
    df_points["hist_year"] = df_points["hist_year"].astype(int)
    df_points["hist_month"] = df_points["hist_month"].astype(int)
    df_points["hist_day"] = df_points["hist_day"].astype(int)
    df_points["T_ref_n"] = df_points["T_ref_n"].astype(int)

    return df_points


# =============================================================================
# FIGURE 1 NEW : NEMO (mers)
# =============================================================================

def _plot_nemo_new_on_ax_sea(ax, dec0, dec1, sea_name,
                            df_points, recent_provider: NemoRecentProvider,
                            depth_min, depth_max):
    sea_conf = SEAS[sea_name]
    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)
    x_for_xlim = []

    df_period = df_points[(df_points['hist_year'] >= dec0) & (df_points['hist_year'] < dec1)].copy()
    df_sea = filter_by_sea(df_period, sea_conf, lon_col='hist_lon', lat_col='hist_lat')

    if len(df_sea) < 1:
        ax.text(0.5, 0.5, "Pas assez de données",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color='gray')
        ax.set_ylim(y_bottom, y_top)
        ax.grid(True, alpha=0.25)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        return x_for_xlim

    # Points historiques (uniques)
    ax.scatter(df_sea['nemo_hist_T'], df_sea['hist_depth_m'],
               c='steelblue', s=12, alpha=0.4, zorder=2)
    x_for_xlim.append(df_sea['nemo_hist_T'].values)

    # Courbe/bandes récentes depuis NetCDF (grille complète)
    depths_i, mu, s1 = recent_provider.recent_profile_sea(sea_name)
    ok = np.isfinite(mu) & np.isfinite(s1)

    if ok.any():
        ax.plot(mu[ok], depths_i[ok], c='black', lw=1.5, zorder=7)
        ax.fill_betweenx(depths_i[ok], (mu - s1)[ok], (mu + s1)[ok],
                         color='red', alpha=0.25, zorder=6)
        ax.fill_betweenx(depths_i[ok], (mu - 2*s1)[ok], (mu + 2*s1)[ok],
                         color='red', alpha=0.15, zorder=5)

        x_for_xlim.append(mu[ok])
        x_for_xlim.append((mu - 2*s1)[ok])
        x_for_xlim.append((mu + 2*s1)[ok])

    ax.set_ylim(y_bottom, y_top)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.25)

    return x_for_xlim


def plot_1_nemo_new_seas_panels(df_points, recent_provider,
                               out_dir, prefix,
                               depth_min=10, depth_max=200):
    periods = [(1900, 1930, "1900-1929"), (1930, 1970, "1930-1969")]
    order = ['Bellingshausen-Amundsen', 'Weddell', 'Davis', 'Ross']

    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor='steelblue', markeredgecolor='none',
               label='Température historique'),
        Line2D([0], [0], color='black', lw=1.8, label='Température récente'),
        Patch(facecolor='red', alpha=0.22, label='±1σ'),
        Patch(facecolor='red', alpha=0.12, label='±2σ'),
    ]

    fig, axes = plt.subplots(
        nrows=len(order), ncols=len(periods),
        figsize=(12, 12),
        sharex=True,
        sharey=True
    )

    xvals_by_col = {j: [] for j in range(len(periods))}

    for i, sea_name in enumerate(order):
        for j, (dec0, dec1, lab) in enumerate(periods):
            ax = axes[i, j]
            x_for_xlim = _plot_nemo_new_on_ax_sea(
                ax, dec0, dec1, sea_name,
                df_points, recent_provider,
                depth_min, depth_max
            )
            xvals_by_col[j].extend(x_for_xlim)

            if i == len(order) - 1:
                ax.set_xlabel("Température (°C)")
            if j == 0:
                ax.set_ylabel("Profondeur (m)")
            else:
                ax.set_ylabel("")

    for j, (_, _, lab) in enumerate(periods):
        axes[0, j].set_title(lab, fontsize=11, fontweight='bold')

    for i, sea_name in enumerate(order):
        sea_label = SEAS[sea_name].get('label', sea_name)
        row_ax = axes[i, 0]
        row_ax.text(-0.25, 0.5, sea_label,
                    transform=row_ax.transAxes,
                    ha='right', va='center',
                    fontsize=10, fontweight='bold')

    for ax in axes.ravel():
        ax.tick_params(labelbottom=True, labelleft=True)

    xvals_union = []
    for j in range(len(periods)):
        xvals_union.extend(xvals_by_col[j])

    xmin_raw, xmax_raw = _compute_xlim_from_content(xvals_union, pad_abs=0.25, pad_frac=0.05)

    tick_min = int(np.ceil(xmin_raw))
    tick_max = int(np.floor(xmax_raw))
    if tick_min > tick_max:
        tick_min = int(np.floor(xmin_raw))
        tick_max = int(np.ceil(xmax_raw))

    xmin, xmax = tick_min - 0.5, tick_max + 0.5
    axes[0, 0].set_xlim(xmin, xmax)
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(1.0))

    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)
    for ax in axes.ravel():
        ax.set_ylim(y_bottom, y_top)

    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               fontsize=LEGEND_FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, LEGEND_Y))

    fig.subplots_adjust(top=0.95, bottom=0.09, left=0.22,
                        hspace=0.25, wspace=0.15)

    out_path = out_dir / f"{prefix}_1NEW_nemo_seas_1900-1929_1930-1969.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Panel NEW mers: {out_path.name}")


# =============================================================================
# FIGURE 1 OLD : latbands 7×2 (1900–1969)
# =============================================================================

def _plot_nemo_latband_on_ax(ax, dec0, dec1, lat_min, lat_max,
                            df_points, recent_provider,
                            depth_min, depth_max,
                            sea_name=None):
    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)
    x_for_xlim = []

    sub = df_points[
        (df_points['hist_year'] >= dec0) &
        (df_points['hist_year'] < dec1) &
        (df_points['hist_lat'] >= -lat_max) &
        (df_points['hist_lat'] < -lat_min) &
        (df_points['hist_depth_m'] >= depth_min) &
        (df_points['hist_depth_m'] <= depth_max)
    ].copy()

    if sea_name is not None:
        sub = filter_by_sea(sub, SEAS[sea_name], lon_col='hist_lon', lat_col='hist_lat')

    if len(sub) < 1:
        ax.text(0.5, 0.5, "Pas assez de données",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color='gray')
        ax.set_ylim(y_bottom, y_top)
        ax.grid(True, alpha=0.25)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        return x_for_xlim

    ax.scatter(sub['nemo_hist_T'], sub['hist_depth_m'],
               c='steelblue', s=12, alpha=0.4, zorder=2)
    x_for_xlim.append(sub['nemo_hist_T'].values)

    depths_i, mu, s1 = recent_provider.recent_profile_latband(lat_min, lat_max, sea_name)
    ok = np.isfinite(mu) & np.isfinite(s1)

    if ok.any():
        ax.plot(mu[ok], depths_i[ok], c='black', lw=1.5, zorder=7)
        ax.fill_betweenx(depths_i[ok], (mu - s1)[ok], (mu + s1)[ok],
                         color='red', alpha=0.25, zorder=6)
        ax.fill_betweenx(depths_i[ok], (mu - 2*s1)[ok], (mu + 2*s1)[ok],
                         color='red', alpha=0.15, zorder=5)

        x_for_xlim.append(mu[ok])
        x_for_xlim.append((mu - 2*s1)[ok])
        x_for_xlim.append((mu + 2*s1)[ok])

    ax.set_ylim(y_bottom, y_top)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.25)

    return x_for_xlim


def plot_1_nemo_latband_panels_1900_1969(df_points, recent_provider,
                                        out_dir, prefix,
                                        depth_min=10, depth_max=200,
                                        sea_name=None):
    decades = [(1900, 1910), (1910, 1920), (1920, 1930),
               (1930, 1940), (1940, 1950), (1950, 1960), (1960, 1970)]
    lat_ranges = [(60, 70), (70, 80)]

    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor='steelblue', markeredgecolor='none',
               label='Température historique'),
        Line2D([0], [0], color='black', lw=1.8, label='Température récente'),
        Patch(facecolor='red', alpha=0.22, label='±1σ'),
        Patch(facecolor='red', alpha=0.12, label='±2σ'),
    ]

    fig, axes = plt.subplots(nrows=len(decades), ncols=len(lat_ranges),
                             figsize=(12, 16), sharex=True, sharey=True)

    xvals_all = []

    for i, (dec0, dec1) in enumerate(decades):
        for j, (lat_min, lat_max) in enumerate(lat_ranges):
            ax = axes[i, j]
            x_for_xlim = _plot_nemo_latband_on_ax(
                ax, dec0, dec1, lat_min, lat_max,
                df_points, recent_provider,
                depth_min, depth_max,
                sea_name=sea_name
            )
            xvals_all.extend(x_for_xlim)

            if i == 0:
                ax.set_title(f"{lat_min}–{lat_max}°S", fontsize=12, fontweight='bold')

            if j == 0:
                ax.text(-0.20, 0.5, f"{dec0}-{dec1-1}",
                        transform=ax.transAxes,
                        ha='right', va='center',
                        fontsize=11, fontweight='bold')

            if i == len(decades) - 1:
                ax.set_xlabel("Température (°C)")

    for ax in axes.ravel():
        ax.tick_params(labelbottom=True, labelleft=True)

    for i in range(len(decades)):
        axes[i, 0].set_ylabel("Profondeur (m)")

    xmin, xmax = _compute_xlim_from_content(xvals_all, pad_abs=0.25, pad_frac=0.05)
    for ax in axes.ravel():
        ax.set_xlim(xmin, xmax)

    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)
    for ax in axes.ravel():
        ax.set_ylim(y_bottom, y_top)

    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               fontsize=LEGEND_FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, LEGEND_Y))

    fig.subplots_adjust(top=0.96, bottom=0.085,
                        left=0.18, right=0.97,
                        hspace=0.30, wspace=0.20)

    if sea_name is None:
        out_path = out_dir / f"{prefix}_1NEW_nemo_latbands_1900-1969.png"
    else:
        out_path = out_dir / f"{prefix}_1NEW_nemo_latbands_{sea_name}_1900-1969.png"

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Panel latbands 1900-1969: {out_path.name}")


# =============================================================================
# FIGURE 2 : Scatter no_reg (STRICTEMENT point-par-point via CSV)
# =============================================================================

def plot_2_no_reg(df_points_period, out_path, title_str):
    """
    Scatter NEMO_hist vs T_ref (moyenne 2005–2023) + histogrammes.
    Point-par-point via CSV (T_ref déjà dans df_points_period).

    MODIF (demandée) :
    - Histogrammes avec classes fixes de 0,5°C (bins explicites),
      alignées sur des multiples de 0,5.
    - Axes scatter avec une marge "Yamazaki-like" : limites non forcées sur des entiers,
      tout en gardant les bins alignés à 0,5°C.
    - y=x tracé coin-à-coin selon les limites effectives.
    """
    # garder uniquement points avec T_ref calculable
    df_scatter = df_points_period.dropna(subset=["nemo_hist_T", "T_ref"]).copy()
    if df_scatter.empty:
        print("⚠️ Scatter: aucune donnée (nemo_hist_T ou T_ref manquant).")
        return

    fig = plt.figure(figsize=(12, 11))
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.98)

    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05, top=0.95)
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)

    BIN_WIDTH = 0.5

    # -------------------------------------------------------------------------
    # 1) Limites d'axes (marge) calculées sur l'union x/y, comme Yamazaki
    #    + petite règle : éviter des bornes exactement entières.
    # -------------------------------------------------------------------------
    T_min_raw = float(min(df_scatter['nemo_hist_T'].min(), df_scatter['T_ref'].min()))
    T_max_raw = float(max(df_scatter['nemo_hist_T'].max(), df_scatter['T_ref'].max()))

    lim_low = T_min_raw - 0.5
    lim_high = T_max_raw + 0.5

    def _avoid_integer(val: float, direction: str) -> float:
        # direction: "down" (borne basse) ou "up" (borne haute)
        if np.isfinite(val) and np.isclose(val, round(val), atol=1e-12):
            return val - 0.25 if direction == "down" else val + 0.25
        return val

    lim_low = _avoid_integer(lim_low, "down")
    lim_high = _avoid_integer(lim_high, "up")

    # -------------------------------------------------------------------------
    # 2) Bins histogrammes : alignés sur multiples de 0,5°C (indépendants des xlim)
    # -------------------------------------------------------------------------
    bin_start = np.floor(lim_low / BIN_WIDTH) * BIN_WIDTH
    bin_end   = np.ceil(lim_high / BIN_WIDTH) * BIN_WIDTH
    bins_edges = np.arange(bin_start, bin_end + BIN_WIDTH, BIN_WIDTH)

    # -------------------------------------------------------------------------
    # Deux classes profondeur
    # -------------------------------------------------------------------------
    bins_depth = [10, 100, 200.0001]
    labels = ['10-100m', '100-200m']
    colors = ['#1f77b4', '#ff7f0e']

    df_plot = df_scatter.copy()
    df_plot['depth_cat'] = pd.cut(df_plot['hist_depth_m'],
                                  bins=bins_depth, labels=labels,
                                  include_lowest=True, right=False)

    for label, color in zip(labels, colors):
        subset = df_plot[df_plot['depth_cat'] == label]
        if len(subset) > 0:
            ax_scatter.scatter(subset['nemo_hist_T'], subset['T_ref'],
                               c=color, s=50, alpha=0.6,
                               edgecolors='black', linewidths=0.3,
                               label=label, zorder=2)

    # -------------------------------------------------------------------------
    # 3) Forcer limites (non entières) + diagonale y=x coin-à-coin
    # -------------------------------------------------------------------------
    ax_scatter.set_xlim(lim_low, lim_high)
    ax_scatter.set_ylim(lim_low, lim_high)

    ax_scatter.plot([lim_low, lim_high], [lim_low, lim_high],
                    'k--', lw=2, alpha=0.7, label='y=x', zorder=10)

    ax_scatter.set_xlabel('Température Historique (°C)')
    ax_scatter.set_ylabel('Température récente (°C)')
    ax_scatter.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_aspect('equal', adjustable='box')

    # Histogrammes : bins fixes 0,5°C
    ax_histx.hist(df_scatter['nemo_hist_T'], bins=bins_edges,
                  color='steelblue', alpha=0.7, edgecolor='black')
    ax_histx.set_ylabel('Effectifs', fontsize=9)
    ax_histx.tick_params(labelbottom=False)
    ax_histx.grid(True, alpha=0.3, axis='y')

    ax_histy.hist(df_scatter['T_ref'], bins=bins_edges,
                  color='steelblue', alpha=0.7, edgecolor='black',
                  orientation='horizontal')
    ax_histy.set_xlabel('Effectifs', fontsize=9)
    ax_histy.tick_params(labelleft=False)
    ax_histy.grid(True, alpha=0.3, axis='x')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Scatter: {out_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Profils NEMO historique (CSV positions obs) vs récent (NetCDF grille) — aligned Yamazaki DA"
    )
    p.add_argument('--csv-dir', type=str, required=True,
                   help="Dossier CSV NEMO (Datas_OK racine : nemo_yamazaki_<SEASON>_*.csv)")
    p.add_argument('--nemo-nc', type=str, required=True,
                   help="NetCDF NEMO thetao monthly (incl. 2005–2023) sur grille")
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--season', type=str, default='DJF',
                   choices=list(SEASON_MONTHS.keys()))
    p.add_argument('--depth-min', type=float, default=10)
    p.add_argument('--depth-max', type=float, default=200)
    p.add_argument('--hist-y0', type=int, default=1900)
    p.add_argument('--hist-y1', type=int, default=1970)
    p.add_argument('--chunksize', type=int, default=1_000_000)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROFILS NEMO — HIST (CSV Datas_OK) vs RÉCENT (NetCDF) | Scatter = point-par-point CSV")
    print("=" * 80)

    # 1) Charger points uniques + T_ref (point-par-point) depuis CSV
    print("\n📂 Construction des points uniques + T_ref (CSV, 2005–2023) ...")
    df_points = load_unique_points_with_tref(
        args.csv_dir, args.season,
        args.depth_min, args.depth_max,
        args.hist_y0, args.hist_y1,
        chunksize=args.chunksize
    )
    if df_points.empty:
        raise RuntimeError("Aucune donnée points uniques / T_ref construite depuis les CSV.")

    print(f"✓ Points uniques: {len(df_points):,}")
    print(f"✓ T_ref non-nan : {df_points['T_ref'].notna().sum():,}")

    # 2) Provider NetCDF (profils régionaux)
    recent_provider = NemoRecentProvider(
        args.nemo_nc, args.season, args.depth_min, args.depth_max
    )

    prefix = f"nemo_{args.season}_{int(args.depth_min)}-{int(args.depth_max)}m"

    # FIG.1 NEW (mers)
    print("\n📊 Génération Fig.1 NEW NEMO (panneau mers)...")
    plot_1_nemo_new_seas_panels(
        df_points, recent_provider,
        out_dir, prefix,
        args.depth_min, args.depth_max
    )

    # FIG.1 OLD (latbands)
    print("\n📊 Génération Fig.1 OLD NEMO (panneau latbands 1900-1969, 7×2)...")
    plot_1_nemo_latband_panels_1900_1969(
        df_points, recent_provider,
        out_dir, prefix,
        args.depth_min, args.depth_max,
        sea_name=None
    )

    # latbands par mer (optionnel, comme votre version)
    print("\n📊 Génération latbands par mer (Weddell, Bellingshausen-Amundsen)...")
    for sea in ['Weddell', 'Bellingshausen-Amundsen']:
        plot_1_nemo_latband_panels_1900_1969(
            df_points, recent_provider,
            out_dir, prefix,
            args.depth_min, args.depth_max,
            sea_name=sea
        )

    # FIG.2 Scatter (1900-1929)
    print("\n📊 Génération Plot 2 NEMO (scatter 1900-1929) [point-par-point CSV] ...")
    df_1900 = df_points[(df_points["hist_year"] >= 1900) & (df_points["hist_year"] < 1930)].copy()
    title_1900 = f"Saison: {args.season} | Période: 1900-1929"
    plot_2_no_reg(
        df_1900,
        out_dir / f"{prefix}_1900-1929_2_no_reg.png",
        title_1900
    )

    # FIG.2 Scatter (1930-1969)
    print("\n📊 Génération Plot 2 NEMO (scatter 1930-1969) [point-par-point CSV] ...")
    df_1930 = df_points[(df_points["hist_year"] >= 1930) & (df_points["hist_year"] < 1970)].copy()
    title_1930 = f"Saison: {args.season} | Période: 1930-1969"
    plot_2_no_reg(
        df_1930,
        out_dir / f"{prefix}_1930-1969_2_no_reg.png",
        title_1930
    )

    print("\n✅ Génération terminée.")


if __name__ == "__main__":
    main()
