"""
commande d'execution : 
python3 ~/Thesis/Yamazaki/Codes/yamazaki_and_histo_profiles_OK.py \
  --csv-dir ~/Thesis/Yamazaki/Datas/Datas_OK \
  --yamazaki-nc ~/Thesis/Yamazaki/Datas/Yamazaki_SO-Monthly-Climatology_v20240604.nc \
  --out-dir ~/Thesis/Yamazaki/Outputs \
  --season DJF \
  --depth-min 10 --depth-max 200 \
  --hist-y0 1900 --hist-y1 1970

- Pour Yamazaki (temp et temp_dev) : on calcule d'abord les statistiques régionales/saisonnières
  sur les niveaux de profondeur NATIFS, puis on interpole les profils (mu, sigma_RMS) sur une grille 1 m.

- définition RMS (sigma_RMS = sqrt(mean(temp_dev^2)) sur les dims non-depth)

- FIGURE 2 (scatter) :
  (1) histogrammes avec classes fixes de 0,5°C, alignées sur des multiples de 0,5 (comme NEMO)
  (2) droite y=x forcée coin-à-coin via xlim/ylim identiques sur l'axe scatter
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator  
from matplotlib.ticker import MultipleLocator
import xarray as xr

plt.rcParams.update({
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
})

SEASON_MONTHS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'ALL': list(range(1, 13)),
}

# =============================================================================#
# DA / LÉGENDES / SPACING 
# =============================================================================#

LEGEND_FONTSIZE_FIG1_NEW = 12
LEGEND_FONTSIZE_MERGED = 12

# NEW mers (4x2)
LEGEND_Y_FIG1_NEW = 0.020
BOTTOM_FIG1_NEW = 0.090

# MERGED 7x2 (1900–1969) 
LEGEND_Y_MERGED = 0.028
BOTTOM_MERGED   = 0.075

# =============================================================================#
# DÉFINITION DES MERS (0-360° sens horaire)
# =============================================================================#

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

# =============================================================================#
# UTILITAIRES
# =============================================================================#

def _safe_read_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.dropna(how='all').empty:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Lecture échouée: {f} ({e})")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def normalize_longitude_360(lon):
    return lon % 360


def filter_by_sea(df, sea_config):
    df = df.copy()
    df['lon_360'] = normalize_longitude_360(df['hist_lon'])

    df_sea = df[
        (df['hist_lat'] >= sea_config['lat_min']) &
        (df['hist_lat'] < sea_config['lat_max'])
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


def _legend_elements_for_bands():
    # STDDEV uniquement
    return [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor='steelblue', markeredgecolor='none',
               label='Température historique'),
        Line2D([0], [0], color='black', lw=1.8, label='Température récente'),
        Patch(facecolor='red', alpha=0.22, label='±1σ'),
        Patch(facecolor='red', alpha=0.12, label='±2σ'),
    ]


def _rms_over_all_non_depth_dims(da):
    """
    RMS sur toutes les dimensions sauf 'depth' :
        RMS = sqrt( mean( da^2 ) )
    da : DataArray avec dimension 'depth' + autres dims (lat/lon/month ...)

    Retourne un DataArray 1D sur 'depth'.
    """
    other_dims = [d for d in da.dims if d != "depth"]
    return np.sqrt((da ** 2).mean(dim=other_dims, skipna=True))


def _ensure_increasing_depth(da_1d_depth: xr.DataArray) -> xr.DataArray:
    """
    Assure un axe depth monotone croissant (utile pour interp).
    Ne modifie que l'ordre si nécessaire.
    """
    depth = da_1d_depth["depth"].values
    if depth.ndim != 1 or depth.size < 2:
        return da_1d_depth
    if np.all(np.diff(depth) > 0):
        return da_1d_depth
    # tri
    order = np.argsort(depth)
    return da_1d_depth.isel(depth=order)


# =============================================================================#
# CHARGEMENT CSV HISTORIQUES
# =============================================================================#

def load_frames(csv_dir, season, depth_min, depth_max, y0, y1):
    pattern = str(Path(csv_dir) / "yamazaki_en4_wod_DJF_*.csv")
    df_all = _safe_read_csvs(pattern)

    if df_all.empty:
        raise FileNotFoundError(f"Aucun CSV trouvé : {pattern}")

    print(f" {len(df_all):,} lignes chargées")

    num_cols = ['hist_month', 'hist_depth_m', 'hist_year',
                'hist_temperature', 'yamazaki_T', 'hist_lat', 'hist_lon']
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

    df = df_all[df_all['hist_month'].isin(SEASON_MONTHS[season])].copy()
    df = df[(df['hist_year'] >= y0) & (df['hist_year'] < y1)]

    df['T_ref'] = df['yamazaki_T']

    df = df[
        (df['hist_temperature'].between(-10, 10)) &
        (df['T_ref'].between(-10, 10))
    ]

    df_cmp = df[
        (df['hist_depth_m'] >= max(10, depth_min)) &
        (df['hist_depth_m'] <= depth_max)
    ]

    df_scatter = df[df['hist_depth_m'] >= max(10, depth_min)]

    print(f" {len(df_cmp):,} observations (depth ≤ {depth_max}m)")
    print(f" {len(df_scatter):,} observations (scatter, sans plafond depth)")

    return df_cmp, df_scatter


# =============================================================================#
# YAMAZAKI NETCDF
# =============================================================================#

def open_yamazaki(nc_path: Path):
    ds = None
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(nc_path, engine=eng, decode_times=False)
            break
        except Exception:
            pass

    if ds is None:
        raise RuntimeError(f"Impossible d'ouvrir {nc_path}")

    # STDDEV uniquement
    required = ("temp", "temp_dev")
    for v in required:
        if v not in ds.data_vars:
            raise RuntimeError(f"Variable '{v}' introuvable dans le NetCDF Yamazaki")

    daT = ds["temp"]
    daSTD = ds["temp_dev"]

    def _standardize(da):
        remap = {}
        if "latitude" in da.dims:
            remap["latitude"] = "lat"
        if "longitude" in da.dims:
            remap["longitude"] = "lon"
        if "lev" in da.dims:
            remap["lev"] = "depth"
        if "z" in da.dims:
            remap["z"] = "depth"
        if "time" in da.dims:
            remap["time"] = "month"
        if remap:
            da = da.rename(remap)
        return da

    daT = _standardize(daT)
    daSTD = _standardize(daSTD)

    print(f"[Yamazaki] temp     dims={list(daT.dims)}    shape={tuple(daT.shape)}")
    print(f"[Yamazaki] temp_dev dims={list(daSTD.dims)} shape={tuple(daSTD.shape)}")

    return daT, daSTD


def _apply_lon_mask(sub, lon_start, lon_end):
    lon = sub['lon']
    lon360 = ((lon + 360) % 360)

    if lon_start <= lon_end:
        lon_mask = (lon360 >= lon_start) & (lon360 < lon_end)
    else:
        lon_mask = (lon360 >= lon_start) | (lon360 < lon_end)

    return sub.isel(lon=lon_mask)


# =============================================================================#
# PROFILS YAMAZAKI : STATS NATIVES -> INTERP 1 m (ALIGN NEMO)
# =============================================================================#

def yamazaki_profile_sea(daT, daSTD, season, sea_config,
                         depth_min=10, depth_max=200):
    """
    Aligné sur NEMO:
    1) sous-ensemble région/saison sur grilles natives
    2) calcul mu(depth_native) et sigma_RMS(depth_native)
    3) interpolation des deux profils 1D sur depths = 1 m (entiers)
    """
    months = SEASON_MONTHS[season]
    subT = daT.sel(month=months)
    subS = daSTD.sel(month=months)

    lat_min = sea_config['lat_min']
    lat_max = sea_config['lat_max']
    lat0 = min(lat_min, lat_max)
    lat1 = max(lat_min, lat_max)

    subT = subT.sel(lat=slice(lat0, lat1))
    subS = subS.sel(lat=slice(lat0, lat1))

    subT = _apply_lon_mask(subT, sea_config['lon_start'], sea_config['lon_end'])
    subS = _apply_lon_mask(subS, sea_config['lon_start'], sea_config['lon_end'])

    subT = subT.where(np.isfinite(subT))
    subS = subS.where(np.isfinite(subS))

    # restreindre profondeur SUR NATIF (avant stats) pour éviter d'interpoler hors plage utile
    subT = subT.sel(depth=subT["depth"].where((subT["depth"] >= depth_min) & (subT["depth"] <= depth_max), drop=True))
    subS = subS.sel(depth=subS["depth"].where((subS["depth"] >= depth_min) & (subS["depth"] <= depth_max), drop=True))

    # ---- STATS sur niveaux natifs
    mu_native = subT.mean(dim=[d for d in subT.dims if d != "depth"], skipna=True)
    bb_native = _rms_over_all_non_depth_dims(subS)

    mu_native = _ensure_increasing_depth(mu_native)
    bb_native = _ensure_increasing_depth(bb_native)

    # ---- INTERP 1 m APRES stats
    depths = np.arange(int(np.ceil(depth_min)), int(np.floor(depth_max)) + 1, 1)
    mu_1m = mu_native.interp(depth=depths, kwargs={"fill_value": np.nan})
    bb_1m = bb_native.interp(depth=depths, kwargs={"fill_value": np.nan})

    mu = np.array(mu_1m.values, dtype=float).squeeze()
    bb = np.array(bb_1m.values, dtype=float).squeeze()

    if mu.ndim != 1 or mu.shape[0] != depths.size:
        mu = np.reshape(mu, (depths.size,))
    if bb.ndim != 1 or bb.shape[0] != depths.size:
        bb = np.reshape(bb, (depths.size,))

    return depths, mu, bb


def yamazaki_profile_sea_latband(daT, daSTD, season, sea_config, lat_min_deg, lat_max_deg,
                                 depth_min=10, depth_max=200):
    """
    Même logique que yamazaki_profile_sea, mais avec un latband + mer.
    """
    months = SEASON_MONTHS[season]
    lat0, lat1 = -lat_max_deg, -lat_min_deg
    lat_slice = slice(lat0, lat1) if lat0 < lat1 else slice(lat1, lat0)

    subT = daT.sel(month=months).sel(lat=lat_slice)
    subS = daSTD.sel(month=months).sel(lat=lat_slice)

    subT = _apply_lon_mask(subT, sea_config['lon_start'], sea_config['lon_end'])
    subS = _apply_lon_mask(subS, sea_config['lon_start'], sea_config['lon_end'])

    subT = subT.where(np.isfinite(subT))
    subS = subS.where(np.isfinite(subS))

    subT = subT.sel(depth=subT["depth"].where((subT["depth"] >= depth_min) & (subT["depth"] <= depth_max), drop=True))
    subS = subS.sel(depth=subS["depth"].where((subS["depth"] >= depth_min) & (subS["depth"] <= depth_max), drop=True))

    # STATS natifs
    mu_native = subT.mean(dim=[d for d in subT.dims if d != "depth"], skipna=True)
    bb_native = _rms_over_all_non_depth_dims(subS)

    mu_native = _ensure_increasing_depth(mu_native)
    bb_native = _ensure_increasing_depth(bb_native)

    # INTERP 1 m après stats
    depths = np.arange(int(np.ceil(depth_min)), int(np.floor(depth_max)) + 1, 1)
    mu_1m = mu_native.interp(depth=depths, kwargs={"fill_value": np.nan})
    bb_1m = bb_native.interp(depth=depths, kwargs={"fill_value": np.nan})

    mu = np.array(mu_1m.values, dtype=float).squeeze()
    bb = np.array(bb_1m.values, dtype=float).squeeze()

    if mu.ndim != 1 or mu.shape[0] != depths.size:
        mu = np.reshape(mu, (depths.size,))
    if bb.ndim != 1 or bb.shape[0] != depths.size:
        bb = np.reshape(bb, (depths.size,))

    return depths, mu, bb


def yamazaki_profile_latband_global(daT, daSTD, season, lat_min_deg, lat_max_deg,
                                   depth_min=10, depth_max=200):
    """
    Latband global (sans filtre mer) avec la même logique STATS natifs -> INTERP 1 m.
    """
    months = SEASON_MONTHS[season]
    lat0, lat1 = -lat_max_deg, -lat_min_deg
    lat_slice = slice(lat0, lat1) if lat0 < lat1 else slice(lat1, lat0)

    subT = daT.sel(month=months).sel(lat=lat_slice).where(np.isfinite(daT))
    subS = daSTD.sel(month=months).sel(lat=lat_slice).where(np.isfinite(daSTD))

    subT = subT.sel(depth=subT["depth"].where((subT["depth"] >= depth_min) & (subT["depth"] <= depth_max), drop=True))
    subS = subS.sel(depth=subS["depth"].where((subS["depth"] >= depth_min) & (subS["depth"] <= depth_max), drop=True))

    mu_native = subT.mean(dim=[d for d in subT.dims if d != "depth"], skipna=True)
    bb_native = _rms_over_all_non_depth_dims(subS)

    mu_native = _ensure_increasing_depth(mu_native)
    bb_native = _ensure_increasing_depth(bb_native)

    depths = np.arange(int(np.ceil(depth_min)), int(np.floor(depth_max)) + 1, 1)
    mu_1m = mu_native.interp(depth=depths, kwargs={"fill_value": np.nan})
    bb_1m = bb_native.interp(depth=depths, kwargs={"fill_value": np.nan})

    mu = np.array(mu_1m.values, dtype=float).squeeze()
    bb = np.array(bb_1m.values, dtype=float).squeeze()

    if mu.ndim != 1 or mu.shape[0] != depths.size:
        mu = np.reshape(mu, (depths.size,))
    if bb.ndim != 1 or bb.shape[0] != depths.size:
        bb = np.reshape(bb, (depths.size,))

    return depths, mu, bb


# =============================================================================#
# FIGURE 1 : PAR MER (4x2) — STDDEV ONLY
# =============================================================================#

def _plot_yamazaki_new_on_ax_sea(ax, dec0, dec1, sea_name, df_comp,
                                daT, daSTD, season, depth_min, depth_max):
    sea_conf = SEAS[sea_name]

    df_period = df_comp[
        (df_comp['hist_year'] >= dec0) &
        (df_comp['hist_year'] < dec1)
    ].copy()

    df_sea = filter_by_sea(df_period, sea_conf)

    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)

    x_for_xlim = []

    if len(df_sea) < 1:
        ax.text(0.5, 0.5, "Pas assez de données",
                transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
        ax.set_ylim(y_bottom, y_top)
        ax.grid(True, alpha=0.25)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        return x_for_xlim

    ax.scatter(df_sea['hist_temperature'], df_sea['hist_depth_m'],
               c='steelblue', s=12, alpha=0.4, zorder=2)
    x_for_xlim.append(df_sea['hist_temperature'].values)

    depths, mu, bb = yamazaki_profile_sea(daT, daSTD, season, sea_conf, depth_min, depth_max)

    ok = np.isfinite(mu)
    if ok.any():
        ax.plot(mu[ok], depths[ok], c='black', lw=1.5, zorder=7)
        x_for_xlim.append(mu[ok])

    ok2 = np.isfinite(mu) & np.isfinite(bb)
    if ok2.any():
        ax.fill_betweenx(depths[ok2], (mu - bb)[ok2], (mu + bb)[ok2],
                         color='red', alpha=0.25, zorder=6)
        ax.fill_betweenx(depths[ok2], (mu - 2*bb)[ok2], (mu + 2*bb)[ok2],
                         color='red', alpha=0.15, zorder=5)
        x_for_xlim.append((mu - 2*bb)[ok2])
        x_for_xlim.append((mu + 2*bb)[ok2])

    ax.set_ylim(y_bottom, y_top)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.25)

    return x_for_xlim


def plot_1_yamazaki_new_panels(df_comp, daT, daSTD,
                               out_dir, prefix, season,
                               depth_min=10, depth_max=200):
    periods = [
        (1900, 1930, "1900-1929"),
        (1930, 1970, "1930-1969"),
    ]
    order = ['Bellingshausen-Amundsen', 'Weddell', 'Davis', 'Ross']

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
            x_for_xlim = _plot_yamazaki_new_on_ax_sea(
                ax, dec0, dec1, sea_name, df_comp,
                daT, daSTD, season, depth_min, depth_max
            )
            if x_for_xlim is not None:
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
        row_ax.text(
            -0.25, 0.5, sea_label,
            transform=row_ax.transAxes,
            rotation=0,
            ha='right', va='center',
            fontsize=10, fontweight='bold'
        )

    for ax in axes.ravel():
        ax.tick_params(labelbottom=True, labelleft=True)

    # -------------------------------------------------------------------------
    # mêmes xlim + mêmes ticks sur les deux colonnes en prenant 1930–1969 (colonne 1) comme référence.
    # -------------------------------------------------------------------------
    ref_col = 1  # 0 -> 1900-1929, 1 -> 1930-1969
    xmin_raw, xmax_raw = _compute_xlim_from_content(xvals_by_col[ref_col], pad_abs=0.25, pad_frac=0.05)

    # Ticks extrêmes "serrés" (entiers)
    tick_min = int(np.ceil(xmin_raw))
    tick_max = int(np.floor(xmax_raw))

    # Sécurité: si la plage est trop petite/inversée, on retombe sur un arrondi classique
    if tick_min > tick_max:
        tick_min = int(np.floor(xmin_raw))
        tick_max = int(np.ceil(xmax_raw))

    # Élargir uniquement si nécessaire pour ne PAS rogner le contenu (avec marge 0.5)
    if xmin_raw < (tick_min - 0.5):
        tick_min -= 1
    if xmax_raw > (tick_max + 0.5):
        tick_max += 1

    xmin, xmax = tick_min - 0.5, tick_max + 0.5

    # Avec sharex=True, un set_xlim sur l'axe maître suffit (propagé à tous)
    axes[0, 0].set_xlim(xmin, xmax)

    # Ticks entiers uniquement (pas=1)
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(1.0))
    # -------------------------------------------------------------------------

    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)
    for ax in axes.ravel():
        ax.set_ylim(y_bottom, y_top)

    fig.legend(handles=_legend_elements_for_bands(), loc='lower center', ncol=7,
               fontsize=LEGEND_FONTSIZE_FIG1_NEW, frameon=False,
               bbox_to_anchor=(0.5, LEGEND_Y_FIG1_NEW))

    fig.subplots_adjust(top=0.95, bottom=BOTTOM_FIG1_NEW, left=0.22,
                        hspace=0.25, wspace=0.15)

    out_path = out_dir / f"{prefix}_1NEW_yamazaki_seas_1900-1929_1930-1969.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Panel NEW mers (STDDEV RMS): {out_path.name}")


# =============================================================================#
# LATBANDS MERGED 7×2 (1900–1969) — global ou filtré par mer — STDDEV ONLY
# =============================================================================#

def _plot_yamazaki_latband_on_ax(ax, dec0, dec1, lat_min, lat_max,
                                df_comp, daT, daSTD,
                                season, depth_min, depth_max,
                                sea_name=None):
    y_top = 0.0
    y_bottom = float(np.ceil(depth_max) + 10.0)

    x_for_xlim = []

    sub = df_comp[
        (df_comp['hist_year'] >= dec0) &
        (df_comp['hist_year'] < dec1) &
        (df_comp['hist_lat'] >= -lat_max) &
        (df_comp['hist_lat'] < -lat_min) &
        (df_comp['hist_depth_m'] >= depth_min) &
        (df_comp['hist_depth_m'] <= depth_max)
    ].copy()

    if sea_name is not None:
        sub = filter_by_sea(sub, SEAS[sea_name])

    if len(sub) < 1:
        ax.text(0.5, 0.5, "Pas assez de données",
                transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
        ax.set_ylim(y_bottom, y_top)
        ax.grid(True, alpha=0.25)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        return x_for_xlim

    ax.scatter(sub['hist_temperature'], sub['hist_depth_m'],
               c='steelblue', s=12, alpha=0.4, zorder=2)
    x_for_xlim.append(sub['hist_temperature'].values)

    # ---- PROFILS récents Yamazaki : stats natifs -> interp 1 m
    if sea_name is None:
        depths, mu, bb = yamazaki_profile_latband_global(
            daT, daSTD, season, lat_min, lat_max,
            depth_min=depth_min, depth_max=depth_max
        )
    else:
        depths, mu, bb = yamazaki_profile_sea_latband(
            daT, daSTD, season, SEAS[sea_name], lat_min, lat_max,
            depth_min=depth_min, depth_max=depth_max
        )

    ok = np.isfinite(mu)
    if ok.any():
        ax.plot(mu[ok], depths[ok], c='black', lw=1.5, zorder=7)
        x_for_xlim.append(mu[ok])

    ok2 = np.isfinite(mu) & np.isfinite(bb)
    if ok2.any():
        ax.fill_betweenx(depths[ok2], (mu - bb)[ok2], (mu + bb)[ok2],
                         color='red', alpha=0.25, zorder=6)
        ax.fill_betweenx(depths[ok2], (mu - 2*bb)[ok2], (mu + 2*bb)[ok2],
                         color='red', alpha=0.15, zorder=5)
        x_for_xlim.append((mu - 2*bb)[ok2])
        x_for_xlim.append((mu + 2*bb)[ok2])

    ax.set_ylim(y_bottom, y_top)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.25)

    return x_for_xlim


def plot_latbands_merged_panel_1900_1969(df_comp, daT, daSTD, out_dir, prefix, season,
                                        depth_min=10, depth_max=200, sea_name=None):
    decades = [(1900, 1910), (1910, 1920), (1920, 1930),
               (1930, 1940), (1940, 1950), (1950, 1960), (1960, 1970)]
    lat_ranges = [(60, 70), (70, 80)]
    legend_elements = _legend_elements_for_bands()

    sea_tag = ""
    if sea_name is not None:
        sea_tag = "_" + sea_name.replace(" ", "_").replace("/", "-")

    fig, axes = plt.subplots(
        nrows=len(decades), ncols=len(lat_ranges),
        figsize=(12, 18),
        sharex=True, sharey=True
    )

    xvals_all = []

    for i, (dec0, dec1) in enumerate(decades):
        for j, (lat_min, lat_max) in enumerate(lat_ranges):
            ax = axes[i, j]
            x_for_xlim = _plot_yamazaki_latband_on_ax(
                ax, dec0, dec1, lat_min, lat_max,
                df_comp, daT, daSTD,
                season, depth_min, depth_max,
                sea_name=sea_name
            )
            if x_for_xlim is not None:
                xvals_all.extend(x_for_xlim)

            if i == 0:
                ax.set_title(f"{lat_min}–{lat_max}°S", fontsize=12, fontweight='bold')

            if j == 0:
                ax.text(-0.2, 0.5, f"{dec0}-{dec1-1}",
                        transform=ax.transAxes,
                        ha='right', va='center', fontsize=11, fontweight='bold')

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
               fontsize=LEGEND_FONTSIZE_MERGED, frameon=False,
               bbox_to_anchor=(0.5, LEGEND_Y_MERGED))

    fig.subplots_adjust(top=0.98, bottom=BOTTOM_MERGED, left=0.18, right=0.97,
                        hspace=0.30, wspace=0.20)

    out_path = out_dir / f"{prefix}_1NEW_yamazaki_latbands_1900-1969{sea_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f" Latbands 1900-1969 MERGED (STDDEV RMS{'' if sea_name is None else ', sea=' + sea_name}): {out_path.name}")


# =============================================================================#
# FIGURE 2 : Scatter (histogrammes alignés sur NEMO : bins 0,5°C)
# =============================================================================#

def plot_2_no_reg(df_scatter, out_path, title_str):
    fig = plt.figure(figsize=(12, 11))
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.98)

    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05, top=0.95)
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)

    # -------------------------------------------------------------------------
    # Limites communes (et fixes) pour forcer y=x coin-à-coin.
    # On prend l'union des valeurs (hist + ref) + padding 0.5°C
    # -------------------------------------------------------------------------
    _x = pd.to_numeric(df_scatter['hist_temperature'], errors='coerce').to_numpy(dtype=float)
    _y = pd.to_numeric(df_scatter['T_ref'], errors='coerce').to_numpy(dtype=float)
    _x = _x[np.isfinite(_x)]
    _y = _y[np.isfinite(_y)]

    if _x.size == 0 or _y.size == 0:
        # cas pathologique : on sauvegarde une figure vide mais valide
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    lim_low = float(min(_x.min(), _y.min()) - 0.5)
    lim_high = float(max(_x.max(), _y.max()) + 0.5)

    # -------------------------------------------------------------------------
    # Catégories profondeur
    # -------------------------------------------------------------------------
    bins = [10, 100, 200.0001]
    labels = ['10-100m', '100-200m']
    colors = ['#1f77b4', '#ff7f0e']

    df_plot = df_scatter.copy()
    df_plot['depth_cat'] = pd.cut(df_plot['hist_depth_m'],
                                  bins=bins, labels=labels,
                                  include_lowest=True, right=False)

    for label, color in zip(labels, colors):
        subset = df_plot[df_plot['depth_cat'] == label]
        if len(subset) > 0:
            ax_scatter.scatter(subset['hist_temperature'], subset['T_ref'],
                               c=color, s=50, alpha=0.6,
                               edgecolors='black', linewidths=0.3,
                               label=label, zorder=2)

    # -------------------------------------------------------------------------
    # FORCER les limites AVANT de tracer y=x
    # -> garantit une diagonale allant d'un coin à l'autre
    # -------------------------------------------------------------------------
    ax_scatter.set_xlim(lim_low, lim_high)
    ax_scatter.set_ylim(lim_low, lim_high)

    ax_scatter.plot([lim_low, lim_high], [lim_low, lim_high],
                    'k--', lw=2, alpha=0.7, label='y=x', zorder=10)

    ax_scatter.set_xlabel('Température Historique (°C)')
    ax_scatter.set_ylabel('Température Récente (°C)')
    ax_scatter.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_aspect('equal', adjustable='box')

    # -------------------------------------------------------------------------
    # Histogrammes : bins fixes 0,5°C alignées sur multiples de 0,5
    # (on se base sur les mêmes limites que le scatter)
    # -------------------------------------------------------------------------
    BIN_WIDTH = 0.5
    bin_start = np.floor(lim_low / BIN_WIDTH) * BIN_WIDTH
    bin_end   = np.ceil(lim_high / BIN_WIDTH) * BIN_WIDTH
    bins_edges = np.arange(bin_start, bin_end + BIN_WIDTH, BIN_WIDTH)

    ax_histx.hist(df_scatter['hist_temperature'], bins=bins_edges,
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


# =============================================================================#
# MAIN
# =============================================================================#

def main():
    p = argparse.ArgumentParser(
        description="Profils Yamazaki vs historiques (v10e_RMS - STDDEV only) | STATS natifs -> interp 1 m"
    )
    p.add_argument('--csv-dir', type=str, required=True,
                   help="Dossier CSV yamazaki_en4_wod_DJF_*.csv")
    p.add_argument('--yamazaki-nc', type=str, required=True,
                   help="NetCDF Yamazaki")
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--season', type=str, default='DJF',
                   choices=list(SEASON_MONTHS.keys()))
    p.add_argument('--depth-min', type=float, default=10)
    p.add_argument('--depth-max', type=float, default=200)
    p.add_argument('--hist-y0', type=int, default=1900)
    p.add_argument('--hist-y1', type=int, default=1970)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROFILS YAMAZAKI VS HISTORIQUES (STDDEV RMS only) | STATS natifs -> interp 1 m")
    print("=" * 80)

    df_cmp, _ = load_frames(
        args.csv_dir, args.season, args.depth_min, args.depth_max,
        args.hist_y0, args.hist_y1
    )

    daT, daSTD = open_yamazaki(Path(args.yamazaki_nc))

    prefix = f"plot_{args.season}_{int(args.depth_min)}-{int(args.depth_max)}m"

    # Fig.1 NEW par mer : STDDEV uniquement
    print("\n Fig.1 mers (STDDEV RMS)...")
    plot_1_yamazaki_new_panels(df_cmp, daT, daSTD, out_dir, prefix, args.season,
                               args.depth_min, args.depth_max)

    # Latbands MERGED 1900-1969 : global STDDEV uniquement
    print("\n Latbands merged 1900-1969 (STDDEV RMS)...")
    plot_latbands_merged_panel_1900_1969(df_cmp, daT, daSTD, out_dir, prefix, args.season,
                                         args.depth_min, args.depth_max, sea_name=None)

    # Latbands MERGED 1900-1969 : Bellingshausen-Amundsen + Weddell (STDDEV RMS uniquement)
    for sea in ("Bellingshausen-Amundsen", "Weddell"):
        print(f"\n Latbands merged mer={sea} 1900-1969 (STDDEV RMS)...")
        plot_latbands_merged_panel_1900_1969(df_cmp, daT, daSTD, out_dir, prefix, args.season,
                                             args.depth_min, args.depth_max, sea_name=sea)

    # Scatter
    print("\n Plot 2 (scatter 1900-1929)...")
    _, df_scatter_1900 = load_frames(args.csv_dir, args.season,
                                    args.depth_min, args.depth_max, 1900, 1930)
    plot_2_no_reg(df_scatter_1900,
                  out_dir / f"{prefix}_1900-1929_2_no_reg.png",
                  f"Saison: {args.season} | Période: 1900-1929")

    print("\n Plot 2 (scatter 1930-1969)...")
    _, df_scatter_1930 = load_frames(args.csv_dir, args.season,
                                    args.depth_min, args.depth_max, 1930, 1970)
    plot_2_no_reg(df_scatter_1930,
                  out_dir / f"{prefix}_1930-1969_2_no_reg.png",
                  f"Saison: {args.season} | Période: 1930-1969")

    print("\n Génération terminée.")


if __name__ == "__main__":
    main()
