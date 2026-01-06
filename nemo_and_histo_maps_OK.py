"""
commande d'execution :
python3 ~/Thesis/NEMO/Codes/nemo_and_histo_maps_OK.py \
  --csv-dir ~/Thesis/NEMO/Datas/Datas_OK \
  --out-dir ~/Thesis/NEMO/Outputs \
  --season DJF

Générateur de cartes polaires antarctiques pour ΔT NEMO vs NEMO :
    ΔT = nemo_hist_T - nemo_recent_mean(2005–2023)  (définition "théorique")

Fonctionnalités :
- Lecture optimisée par décennie (un fichier à la fois).
- CSV Datas_OK (racine) : fichiers de type
      nemo_yamazaki_DJF_1900_1909.csv
      nemo_yamazaki_DJF_1910_1919.csv
      ...
- Les CSV contiennent 1 obs historique × 19 années récentes (recent_year, nemo_recent_T).
  => On regroupe par observation historique et on moyenne uniquement recent_year 2005–2023.
- Cartes individuelles 2°×2°.
- Panels 1900–1929 (3×2), 1930–1969 (4×2), fusion 1900–1929 vs 1930–1969 (2×2).
- Fusion OOM-safe: accumulation sur grille (sum & count), sans concat gigantesque.

Colonnes nécessaires :
    hist_year, hist_month, hist_day, hist_lat, hist_lon,
    hist_depth_m, nemo_hist_T,
    recent_year, nemo_recent_T
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import gc


# -----------------------------------------------------------------------------#
# Constantes
# -----------------------------------------------------------------------------#

DECADES = [
    (1900, 1909, "1900-1909"),
    (1910, 1919, "1910-1919"),
    (1920, 1929, "1920-1929"),
    (1930, 1939, "1930-1939"),
    (1940, 1949, "1940-1949"),
    (1950, 1959, "1950-1959"),
    (1960, 1969, "1960-1969"),
]

DEPTH_RANGES = [
    (10, 99.99, "10-100m"),
    (100, 200, "100-200m"),
]

SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ALL": list(range(1, 13)),
}

RECENT_Y0 = 2005
RECENT_Y1 = 2023


# -----------------------------------------------------------------------------#
# Outils carto
# -----------------------------------------------------------------------------#

def circle_boundary_path():
    theta = np.linspace(0, 2 * np.pi, 256)
    vertices = np.vstack(
        [0.5 + 0.5 * np.cos(theta), 0.5 + 0.5 * np.sin(theta)]
    ).T
    return MplPath(vertices)


def create_polar_map(ax, lat_limit=-60):
    ax.set_extent([-180, 180, -90, lat_limit], crs=ccrs.PlateCarree())
    ax.set_boundary(circle_boundary_path(), transform=ax.transAxes)
    ax.set_facecolor("#E0E0E0")
    ax.add_feature(
        cfeature.LAND, facecolor="antiquewhite",
        edgecolor="none", zorder=3
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.5, color="black", zorder=4
    )
    ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, lat_limit + 1, 10),
        linewidth=0.5, color="gray", alpha=0.5,
        linestyle="--", zorder=5
    )
    return ax


# -----------------------------------------------------------------------------#
# Lecture / traitements
# -----------------------------------------------------------------------------#

def _decade_csv_path(csv_dir: Path, season: str, decade_label: str) -> Path:
    """
    Vos fichiers Datas_OK suivent le pattern:
      nemo_yamazaki_<SEASON>_<YYYY>_<YYYY>.csv

    Or decade_label dans DECADES est "1900-1909".
    => conversion en "1900_1909"
    """
    lab_u = decade_label.replace("-", "_")
    return csv_dir / f"nemo_yamazaki_{season}_{lab_u}.csv"


def load_decade(csv_dir, y_start, y_end, season, chunksize=1_000_000):
    """
    Charge une décennie complète en mémoire :
    1) lire (hist x recent_year) par chunks
    2) filtrer saison + lat<=-60 + recent_year 2005–2023 + NaNs
    3) grouper par observation historique (point)
    4) nemo_recent_mean = moyenne sur les 19 valeurs (2005–2023)
    5) delta_T = nemo_hist_T - nemo_recent_mean (convention existante)
    """
    decade_label = f"{y_start}-{y_end}"
    csv_file = _decade_csv_path(Path(csv_dir), season, decade_label)

    if not csv_file.exists():
        print(f"    Fichier manquant : {csv_file.name}")
        return pd.DataFrame()

    print(f"    Chargement {csv_file.name}...", end=" ", flush=True)

    season_months = SEASON_MONTHS[season]
    chunks = []

    usecols = [
        "hist_year", "hist_month", "hist_day",
        "hist_lat", "hist_lon",
        "hist_depth_m",
        "nemo_hist_T",
        "recent_year",
        "nemo_recent_T",
    ]

    for chunk in pd.read_csv(csv_file, chunksize=chunksize, usecols=usecols):
        # coercitions (sécurité)
        for c in ["hist_year", "hist_month", "hist_day", "recent_year"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        for c in ["hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T", "nemo_recent_T"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # filtres
        chunk = chunk[chunk["hist_month"].isin(season_months)]
        chunk = chunk[chunk["hist_lat"] <= -60.0]

        # fenêtre récente strictement 2005–2023
        chunk = chunk[(chunk["recent_year"] >= RECENT_Y0) & (chunk["recent_year"] <= RECENT_Y1)]

        # nettoyage NaN
        chunk = chunk[
            chunk["hist_lat"].notna()
            & chunk["hist_lon"].notna()
            & chunk["hist_depth_m"].notna()
            & chunk["nemo_hist_T"].notna()
            & chunk["nemo_recent_T"].notna()
        ]

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        print("0 obs")
        return pd.DataFrame()

    df_all = pd.concat(chunks, ignore_index=True)

    # Grouper par observation historique (point)
    group_cols = ["hist_year", "hist_month", "hist_day", "hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T"]
    df_grouped = df_all.groupby(group_cols, as_index=False).agg(
        nemo_recent_mean=("nemo_recent_T", "mean"),
        recent_n=("nemo_recent_T", "count"),
    )

    # delta_T : conserver votre convention existante
    df_grouped["delta_T"] = df_grouped["nemo_hist_T"] - df_grouped["nemo_recent_mean"]

    print(f"{len(df_grouped):,} obs historiques uniques")
    return df_grouped


def filter_depth(df, depth_min, depth_max):
    return df[
        (df["hist_depth_m"] >= depth_min) &
        (df["hist_depth_m"] <= depth_max)
    ].copy()


# -----------------------------------------------------------------------------#
# Grille 2°×2° pour un DataFrame en mémoire (pour une décennie)
# -----------------------------------------------------------------------------#

def compute_grid_mean(df, grid_size=2.0, min_obs_per_cell=1):
    lat_bins = np.arange(-90, -59, grid_size)
    lon_bins = np.arange(-180, 181, grid_size)
    n_lat, n_lon = len(lat_bins) - 1, len(lon_bins) - 1

    mean_grid = np.full((n_lat, n_lon), np.nan)

    # version directe (boucles) conservée
    for i in range(n_lat):
        for j in range(n_lon):
            mask = (
                (df["hist_lat"] >= lat_bins[i]) &
                (df["hist_lat"] < lat_bins[i + 1]) &
                (df["hist_lon"] >= lon_bins[j]) &
                (df["hist_lon"] < lon_bins[j + 1])
            )
            if mask.sum() >= min_obs_per_cell:
                mean_grid[i, j] = df.loc[mask, "delta_T"].mean()

    return mean_grid, lat_bins, lon_bins


# -----------------------------------------------------------------------------#
# Grille 2°×2° pour un ensemble de décennies (fusion) — version OOM-safe
# -----------------------------------------------------------------------------#

def compute_fused_grid_period(csv_dir, season, depth_min, depth_max,
                              decades_subset, grid_size=2.0,
                              chunksize=1_000_000):
    """
    Fusion OOM-safe :
    - pour chaque décennie : on lit, on regroupe en points uniques (mean 2005–2023),
      puis on accumule delta sur la grille (sum & count).
    """
    lat_bins = np.arange(-90, -59, grid_size)
    lon_bins = np.arange(-180, 181, grid_size)
    n_lat, n_lon = len(lat_bins) - 1, len(lon_bins) - 1

    sum_grid = np.zeros((n_lat, n_lon), dtype=np.float64)
    cnt_grid = np.zeros((n_lat, n_lon), dtype=np.int64)

    season_months = SEASON_MONTHS[season]

    for y0, y1, label in decades_subset:
        csv_file = _decade_csv_path(Path(csv_dir), season, label)
        if not csv_file.exists():
            print(f"    (fusion) Fichier manquant : {csv_file.name}")
            continue

        print(f"    (fusion {label}) lecture par chunks...")

        all_chunks = []
        usecols = [
            "hist_year", "hist_month", "hist_day",
            "hist_lat", "hist_lon",
            "hist_depth_m",
            "nemo_hist_T",
            "recent_year",
            "nemo_recent_T",
        ]

        for chunk in pd.read_csv(csv_file, chunksize=chunksize, usecols=usecols):
            for c in ["hist_year", "hist_month", "hist_day", "recent_year"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            for c in ["hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T", "nemo_recent_T"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            # Saison / domaine
            chunk = chunk[chunk["hist_month"].isin(season_months)]
            chunk = chunk[chunk["hist_lat"] <= -60.0]

            # fenêtre récente 2005–2023
            chunk = chunk[(chunk["recent_year"] >= RECENT_Y0) & (chunk["recent_year"] <= RECENT_Y1)]

            # Profondeur
            chunk = chunk[
                (chunk["hist_depth_m"] >= depth_min) &
                (chunk["hist_depth_m"] <= depth_max)
            ]

            # Nettoyage NaN
            chunk = chunk[
                chunk["hist_lat"].notna()
                & chunk["hist_lon"].notna()
                & chunk["nemo_hist_T"].notna()
                & chunk["nemo_recent_T"].notna()
            ]

            if not chunk.empty:
                all_chunks.append(chunk)

        if not all_chunks:
            continue

        df_decade = pd.concat(all_chunks, ignore_index=True)

        group_cols = ["hist_year", "hist_month", "hist_day", "hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T"]
        df_grouped = df_decade.groupby(group_cols, as_index=False).agg(
            nemo_recent_mean=("nemo_recent_T", "mean")
        )

        # delta : conserver votre convention
        delta = (df_grouped["nemo_hist_T"].to_numpy() - df_grouped["nemo_recent_mean"].to_numpy())
        lats = df_grouped["hist_lat"].to_numpy()
        lons = df_grouped["hist_lon"].to_numpy()

        lat_idx = np.digitize(lats, lat_bins) - 1
        lon_idx = np.digitize(lons, lon_bins) - 1

        valid = (
            (lat_idx >= 0) & (lat_idx < n_lat) &
            (lon_idx >= 0) & (lon_idx < n_lon) &
            np.isfinite(delta)
        )

        if not np.any(valid):
            continue

        lat_idx = lat_idx[valid]
        lon_idx = lon_idx[valid]
        delta_v = delta[valid]

        np.add.at(sum_grid, (lat_idx, lon_idx), delta_v)
        np.add.at(cnt_grid, (lat_idx, lon_idx), 1)

        del df_decade, df_grouped, all_chunks
        gc.collect()

    mean_grid = np.full_like(sum_grid, np.nan, dtype=float)
    mask = cnt_grid > 0
    mean_grid[mask] = sum_grid[mask] / cnt_grid[mask]

    return mean_grid, lat_bins, lon_bins


# -----------------------------------------------------------------------------#
# Plot gridded (décennie ou fusion)
# -----------------------------------------------------------------------------#

def plot_gridded_2deg(df, out_path, season, depth_str, period_str):
    mean_grid, lat_bins, lon_bins = compute_grid_mean(df, grid_size=2.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    create_polar_map(ax, lat_limit=-60)

    LON, LAT = np.meshgrid(lon_bins, lat_bins)
    mesh = ax.pcolormesh(
        LON, LAT, mean_grid,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=-3.5, vmax=3.5,
        shading="flat", zorder=2
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.7)
    cbar.set_label("ΔT (°C)", fontsize=12, fontweight="bold")

    title = "Anomalie de température (2.0° × 2.0°)\n"
    title += f"({period_str}) | {season} | {depth_str}\n"
    title += f"Valid Cells: {np.sum(~np.isnan(mean_grid))}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"       {out_path.name}")


def plot_gridded_2deg_on_axis(ax, csv_dir, season, depth_min, depth_max,
                             depth_str, year_start, year_end, period_label,
                             vmin=-3.5, vmax=3.5):
    df = load_decade(csv_dir, year_start, year_end, season)
    if df.empty:
        ax.text(0.5, 0.5, "Pas de\ndonnées",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    df = filter_depth(df, depth_min, depth_max)

    if df.empty or "delta_T" not in df.columns:
        ax.text(0.5, 0.5, "Pas de\ndonnées",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    if len(df) < 10:
        ax.text(0.5, 0.5, "Pas assez\nd'observations",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    mean_grid, lat_bins, lon_bins = compute_grid_mean(df, grid_size=2.0)
    create_polar_map(ax, lat_limit=-60)

    LON, LAT = np.meshgrid(lon_bins, lat_bins)
    mesh = ax.pcolormesh(
        LON, LAT, mean_grid,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )

    del df, mean_grid
    gc.collect()

    return mesh


# -----------------------------------------------------------------------------#
# Panneaux décénnaux + fusion
# -----------------------------------------------------------------------------#

def create_decadal_panels(csv_dir, out_dir, season):
    out_dir = Path(out_dir)
    vmin, vmax = -3.5, 3.5

    d1_min, d1_max, d1_str = DEPTH_RANGES[0]
    d2_min, d2_max, d2_str = DEPTH_RANGES[1]

    # ---------------- Panel 1 : 1900-1929 (3×2) ----------------
    print("\n Panel 1: 1900-1929")
    panel1_periods = DECADES[:3]

    fig1, axes1 = plt.subplots(
        nrows=3, ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 12)
    )
    axes1 = np.atleast_2d(axes1)

    meshes = []
    for row, (y0, y1, label) in enumerate(panel1_periods):
        m1 = plot_gridded_2deg_on_axis(
            axes1[row, 0], csv_dir, season,
            d1_min, d1_max, d1_str, y0, y1, label,
            vmin, vmax
        )
        m2 = plot_gridded_2deg_on_axis(
            axes1[row, 1], csv_dir, season,
            d2_min, d2_max, d2_str, y0, y1, label,
            vmin, vmax
        )
        if m1 is not None:
            meshes.append(m1)
        if m2 is not None:
            meshes.append(m2)

        axes1[row, 0].text(
            -0.10, 0.5, label,
            transform=axes1[row, 0].transAxes,
            ha="right", va="center",
            fontsize=11, fontweight="bold"
        )

    axes1[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes1[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    if meshes:
        sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="RdBu_r")
        sm.set_array([])
        cax = fig1.add_axes([0.2, 0.10, 0.6, 0.025])
        cbar = fig1.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg = fig1.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg.axis("off")
    rect = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0",
                     edgecolor="black", linewidth=0.5)
    ax_leg.add_patch(rect)
    ax_leg.text(1.05, 0.5, "No Data", va="center",
                ha="left", fontsize=9)

    fig1.suptitle(
        f"Anomalie de température (2°×2°) – {season}",
        fontsize=13, fontweight="bold", y=0.97
    )
    fig1.subplots_adjust(
        left=0.16, right=0.95, top=0.9, bottom=0.14,
        hspace=0.15, wspace=0.05
    )
    p1 = out_dir / f"NEMOvsNEMO_panel_gridded_2deg_{season}_1900-1929_3x2.png"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)
    print(f" {p1.name}")

    # ---------------- Panel 2 : 1930-1969 (4×2) ----------------
    print("\n Panel 2: 1930-1969")
    panel2_periods = DECADES[3:7]

    fig2, axes2 = plt.subplots(
        nrows=4, ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 14)
    )
    axes2 = np.atleast_2d(axes2)

    meshes = []
    for row, (y0, y1, label) in enumerate(panel2_periods):
        m1 = plot_gridded_2deg_on_axis(
            axes2[row, 0], csv_dir, season,
            d1_min, d1_max, d1_str, y0, y1, label,
            vmin, vmax
        )
        m2 = plot_gridded_2deg_on_axis(
            axes2[row, 1], csv_dir, season,
            d2_min, d2_max, d2_str, y0, y1, label,
            vmin, vmax
        )
        if m1 is not None:
            meshes.append(m1)
        if m2 is not None:
            meshes.append(m2)

        axes2[row, 0].text(
            -0.10, 0.5, label,
            transform=axes2[row, 0].transAxes,
            ha="right", va="center",
            fontsize=11, fontweight="bold"
        )

    axes2[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes2[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    if meshes:
        sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="RdBu_r")
        sm.set_array([])
        cax = fig2.add_axes([0.2, 0.10, 0.6, 0.025])
        cbar = fig2.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg = fig2.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg.axis("off")
    rect = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0",
                     edgecolor="black", linewidth=0.5)
    ax_leg.add_patch(rect)
    ax_leg.text(1.05, 0.5, "No Data", va="center",
                ha="left", fontsize=9)

    fig2.suptitle(
        f"Anomalie de température (2°×2°) – {season}",
        fontsize=13, fontweight="bold", y=0.97
    )
    fig2.subplots_adjust(
        left=0.16, right=0.95, top=0.9, bottom=0.14,
        hspace=0.15, wspace=0.05
    )
    p2 = out_dir / f"NEMOvsNEMO_panel_gridded_2deg_{season}_1930-1969_4x2.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)
    print(f"✅ {p2.name}")

    # ---------------- Panel 3 : fusion 1900-1929 vs 1930-1969 (2×2) ----------------
    print("\n Panel 3: 1900-1929 / 1930-1969 (fusion)")

    early_decades = DECADES[:3]
    late_decades  = DECADES[3:7]

    fig3, axes3 = plt.subplots(
        nrows=2, ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 10)
    )
    axes3 = np.atleast_2d(axes3)

    # Ligne 0 : 1900-1929
    print("   Fusion 1900-1929...")
    mean_early_10, lat_bins, lon_bins = compute_fused_grid_period(
        csv_dir, season, d1_min, d1_max, early_decades, grid_size=2.0
    )
    mean_early_100, _, _ = compute_fused_grid_period(
        csv_dir, season, d2_min, d2_max, early_decades, grid_size=2.0
    )

    create_polar_map(axes3[0, 0], lat_limit=-60)
    create_polar_map(axes3[0, 1], lat_limit=-60)

    LON, LAT = np.meshgrid(lon_bins, lat_bins)
    axes3[0, 0].pcolormesh(
        LON, LAT, mean_early_10,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes3[0, 1].pcolormesh(
        LON, LAT, mean_early_100,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )

    axes3[0, 0].text(
        -0.10, 0.5, "1900-1929",
        transform=axes3[0, 0].transAxes,
        ha="right", va="center",
        fontsize=11, fontweight="bold"
    )

    # Ligne 1 : 1930-1969
    print("   Fusion 1930-1969...")
    mean_late_10, _, _ = compute_fused_grid_period(
        csv_dir, season, d1_min, d1_max, late_decades, grid_size=2.0
    )
    mean_late_100, _, _ = compute_fused_grid_period(
        csv_dir, season, d2_min, d2_max, late_decades, grid_size=2.0
    )

    create_polar_map(axes3[1, 0], lat_limit=-60)
    create_polar_map(axes3[1, 1], lat_limit=-60)

    axes3[1, 0].pcolormesh(
        LON, LAT, mean_late_10,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes3[1, 1].pcolormesh(
        LON, LAT, mean_late_100,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )

    axes3[1, 0].text(
        -0.10, 0.5, "1930-1969",
        transform=axes3[1, 0].transAxes,
        ha="right", va="center",
        fontsize=11, fontweight="bold"
    )

    axes3[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes3[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="RdBu_r")
    sm.set_array([])
    cax = fig3.add_axes([0.2, 0.10, 0.6, 0.025])
    cbar = fig3.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg = fig3.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg.axis("off")
    rect = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0",
                     edgecolor="black", linewidth=0.5)
    ax_leg.add_patch(rect)
    ax_leg.text(1.05, 0.5, "No Data", va="center",
                ha="left", fontsize=9)

    fig3.subplots_adjust(
        left=0.16, right=0.95, top=0.95, bottom=0.14,
        hspace=0.15, wspace=0.05
    )
    p3 = out_dir / f"NEMOvsNEMO_panel_gridded_2deg_{season}_1900-1969_2x2.png"
    fig3.savefig(p3, dpi=300)
    plt.close(fig3)
    print(f" {p3.name}")

    gc.collect()


# -----------------------------------------------------------------------------#
# Carte gridded aux positions Yamazaki exactes
# -----------------------------------------------------------------------------#

def create_yamazaki_positions_gridded_map(csv_path, out_dir, season, chunksize=1_000_000):
    """
    Crée une carte gridded 2°×2° aux positions Yamazaki exactes.
    NOTE: ici aussi, on a (hist x 19 recent_year), donc on regroupe avant delta_T.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"    Fichier introuvable : {csv_file}")
        return

    print(f"    Chargement {csv_file.name}...")

    season_months = SEASON_MONTHS[season]
    chunks = []

    usecols = [
        "hist_year", "hist_month", "hist_day",
        "hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T",
        "recent_year", "nemo_recent_T"
    ]

    for chunk in pd.read_csv(csv_file, chunksize=chunksize, usecols=usecols):
        for c in ["hist_year", "hist_month", "hist_day", "recent_year"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        for c in ["hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T", "nemo_recent_T"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        chunk = chunk[chunk["hist_month"].isin(season_months)]
        chunk = chunk[chunk["hist_lat"] <= -60.0]
        chunk = chunk[(chunk["recent_year"] >= RECENT_Y0) & (chunk["recent_year"] <= RECENT_Y1)]

        chunk = chunk[
            chunk["hist_lat"].notna() & chunk["hist_lon"].notna() &
            chunk["hist_depth_m"].notna() & chunk["nemo_hist_T"].notna() &
            chunk["nemo_recent_T"].notna()
        ]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        print("    Aucune donnée")
        return

    df_all = pd.concat(chunks, ignore_index=True)

    group_cols = ["hist_year", "hist_month", "hist_day", "hist_lat", "hist_lon", "hist_depth_m", "nemo_hist_T"]
    df = df_all.groupby(group_cols, as_index=False).agg(
        nemo_recent_mean=("nemo_recent_T", "mean")
    )

    df["delta_T"] = df["nemo_hist_T"] - df["nemo_recent_mean"]
    df = df[df["delta_T"].notna()]

    print(f"   ✓ {len(df):,} observations historiques uniques")

    # Grilles Panel 3
    grid_size = 2.0
    lat_bins = np.arange(-90, -59, grid_size)
    lon_bins = np.arange(-180, 181, grid_size)
    n_lat, n_lon = len(lat_bins) - 1, len(lon_bins) - 1
    LON, LAT = np.meshgrid(lon_bins, lat_bins)

    periods = [(1900, 1929), (1930, 1969)]
    depths = [(10, 99.99), (100, 200)]

    grids = {}
    for (y_start, y_end) in periods:
        for (d_min, d_max) in depths:
            df_sub = df[(df["hist_year"] >= y_start) & (df["hist_year"] <= y_end) &
                        (df["hist_depth_m"] >= d_min) & (df["hist_depth_m"] <= d_max)]

            grid = np.full((n_lat, n_lon), np.nan)
            for i in range(n_lat):
                for j in range(n_lon):
                    mask = ((df_sub["hist_lat"] >= lat_bins[i]) &
                            (df_sub["hist_lat"] < lat_bins[i + 1]) &
                            (df_sub["hist_lon"] >= lon_bins[j]) &
                            (df_sub["hist_lon"] < lon_bins[j + 1]))
                    vals = df_sub.loc[mask, "delta_T"].values
                    if len(vals) > 0:
                        grid[i, j] = np.mean(vals)

            grids[(y_start, y_end, d_min, d_max)] = grid

    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(2, 2, subplot_kw={"projection": ccrs.SouthPolarStereo()})

    vmin, vmax = -3.5, 3.5

    # 1900-1929
    create_polar_map(axes[0, 0], lat_limit=-60)
    create_polar_map(axes[0, 1], lat_limit=-60)

    axes[0, 0].pcolormesh(
        LON, LAT, grids[(1900, 1929, 10, 99.99)],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes[0, 1].pcolormesh(
        LON, LAT, grids[(1900, 1929, 100, 200)],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes[0, 0].text(-0.10, 0.5, "1900-1929",
                    transform=axes[0, 0].transAxes,
                    ha="right", va="center",
                    fontsize=11, fontweight="bold")

    # 1930-1969
    create_polar_map(axes[1, 0], lat_limit=-60)
    create_polar_map(axes[1, 1], lat_limit=-60)

    axes[1, 0].pcolormesh(
        LON, LAT, grids[(1930, 1969, 10, 99.99)],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes[1, 1].pcolormesh(
        LON, LAT, grids[(1930, 1969, 100, 200)],
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )
    axes[1, 0].text(-0.10, 0.5, "1930-1969",
                    transform=axes[1, 0].transAxes,
                    ha="right", va="center",
                    fontsize=11, fontweight="bold")

    axes[0, 0].set_title("10–100 m", fontsize=11, fontweight="bold", pad=10)
    axes[0, 1].set_title("100–200 m", fontsize=11, fontweight="bold", pad=10)

    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap="RdBu_r")
    sm.set_array([])
    cax = fig.add_axes([0.2, 0.10, 0.6, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg = fig.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg.axis("off")
    rect = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0",
                     edgecolor="black", linewidth=0.5)
    ax_leg.add_patch(rect)
    ax_leg.text(1.05, 0.5, "No Data", va="center",
                ha="left", fontsize=9)

    fig.suptitle(
        f"Anomalie de température (2°×2°) – {season}",
        fontsize=13, fontweight="bold", y=0.97
    )
    fig.subplots_adjust(
        left=0.16, right=0.95, top=0.9, bottom=0.14,
        hspace=0.15, wspace=0.05
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"NEMOvsNEMO_yamazaki_positions_gridded_2deg_{season}_panel_2x2.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"    {out_path.name}")
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=str, required=True)
    ap.add_argument("--yamazaki-positions-csv", type=str, default=None,
                    help="CSV avec positions Yamazaki exactes (optionnel)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--season", type=str, default="DJF",
                    choices=list(SEASON_MONTHS.keys()))
    args = ap.parse_args()

    print("=" * 80)
    print("GÉNÉRATEUR NEMO vs NEMO OPTIMISÉ (2°×2°, panels) — CSV Datas_OK")
    print("=" * 80)

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n CARTES INDIVIDUELLES\n")
    for y_start, y_end, period_str in DECADES:
        df = load_decade(csv_dir, y_start, y_end, args.season)
        if df.empty:
            continue
        for depth_min, depth_max, depth_str in DEPTH_RANGES:
            df_d = filter_depth(df, depth_min, depth_max)
            if len(df_d) < 10:
                continue
            out_path = out_dir / (
                f"NEMOvsNEMO_map_{args.season}_{depth_str}_{period_str}_gridded_2deg.png"
            )
            plot_gridded_2deg(df_d, out_path, args.season, depth_str, period_str)
        del df
        gc.collect()

    create_decadal_panels(csv_dir, out_dir, args.season)

    if args.yamazaki_positions_csv:
        print("\n CARTE GRIDDED AUX POSITIONS YAMAZAKI EXACTES\n")
        create_yamazaki_positions_gridded_map(
            args.yamazaki_positions_csv, out_dir, args.season
        )

    print(f"\n{'=' * 80}\n TERMINÉ\n{'=' * 80}")


if __name__ == "__main__":
    main()
