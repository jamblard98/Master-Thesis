"""
commande d'éxécution : 
python3 ~/Thesis/Yamazaki/Codes/yamazaki_and_histo_maps.py \
  --csv-dir ~/Thesis/Yamazaki/Datas/Datas_OK \
  --out-dir ~/Thesis/Yamazaki/Outputs \
  --season DJF

---------------------------
Générateur de cartes polaires antarctiques Yamazaki vs Observations Historiques.

Génère :
- Cartes de localisation des observations
- Cartes griddées 2°×2° (anomalies moyennes)
- Panneaux décennaux :
    * Panel 1 : 1900–1909, 1910–1919, 1920–1929 (3 lignes × 2 profondeurs)
    * Panel 2 : 1930–1939, 1940–1949, 1950–1959, 1960–1969 (4 lignes × 2 profondeurs)
    * Panel 3 : 1900–1929, 1930–1969 (2 lignes × 2 profondeurs)

Inputs  : ~/Thesis/Yamazaki/Datas/Datas_OK/yamazaki_en4_wod_DJF_*.csv
Outputs : ~/Thesis/Yamazaki/Outputs/*.png
"""

import os
import argparse
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.path import Path as MplPath
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# =====================================================================
# CONFIGURATION
# =====================================================================

PERIODS = [
    (1900, 1969, "1900-1969"),
    (1900, 1929, "1900-1929"),
    (1930, 1969, "1930-1969"),
    (1930, 1949, "1930-1949"),
    (1950, 1969, "1950-1969"),
]

# Profondeurs : 10-100m EXCLUT 100, 100-200m INCLUT 100
DEPTH_RANGES = [
    (10, 99.99, "10-100m"),
    (100, 200, "100-200m"),
    (10, 200, "10-200m"),
]

# =====================================================================
# UTILITAIRES CARTE
# =====================================================================

def circle_boundary_path():
    """Crée un path circulaire pour le cadre de la carte."""
    th = np.linspace(0, 2 * np.pi, 256)
    v = np.vstack([0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)]).T
    return MplPath(v)


def create_polar_map(ax, lat_limit=-60):
    """Configure projection stéréographique polaire Sud avec découpe circulaire."""
    ax.set_extent([-180, 180, -90, lat_limit], crs=ccrs.PlateCarree())
    ax.set_boundary(circle_boundary_path(), transform=ax.transAxes)

    ax.set_facecolor("#E0E0E0")

    ax.add_feature(
        cfeature.LAND,
        facecolor="antiquewhite",
        edgecolor="none",
        zorder=3,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.5,
        color="black",
        zorder=4,
    )

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, lat_limit + 1, 10),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=5,
    )

    return ax

# =====================================================================
# CHARGEMENT & FILTRAGE
# =====================================================================

def load_csv_data(csv_dir: Path):
    """Charge TOUS les CSV principaux (exclut _details)."""
    csv_files = sorted(csv_dir.glob("yamazaki_en4_wod_DJF_*.csv"))
    csv_files = [f for f in csv_files if "_details" not in f.name]

    if not csv_files:
        raise ValueError(f" Aucun CSV trouvé dans {csv_dir}")

    print(f" {len(csv_files)} fichiers CSV trouvés")

    dfs = []
    for csv_path in csv_files:
        try:
            # Lire avec na_filter=True pour gérer les valeurs manquantes
            df = pd.read_csv(csv_path, low_memory=False, na_filter=True, keep_default_na=True)
            dfs.append(df)
        except Exception as e:
            print(f"  Erreur lecture {csv_path.name}: {e}")
            continue

    if not dfs:
        raise ValueError(" Aucun CSV valide chargé")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f" {len(df_all):,} observations chargées")
    return df_all


def filter_data(df, season, depth_min, depth_max, year_start, year_end):
    """Filtre les observations (bornes inclusives)."""
    df = df.copy()

    # Saison
    season_months = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    if season in season_months:
        df = df[df["hist_month"].isin(season_months[season])]

    # Années
    df = df[(df["hist_year"] >= year_start) & (df["hist_year"] <= year_end)]

    # Profondeur (bornes inclusives)
    df = df[(df["hist_depth_m"] >= depth_min) & (df["hist_depth_m"] <= depth_max)]

    # Découpage à 60°S
    df = df[df["hist_lat"] <= -60.0]

    # Validité
    df = df[df["hist_lat"].notna() & df["hist_lon"].notna()]
    df = df[df["hist_temperature"].notna()]

    return df


def compute_delta_T(df):
    """Calcule ΔT = T_historique- Yamazaki et filtre NaN."""
    df = df.copy()
    
    # Utiliser yamazaki_T comme référence
    df["ref_T"] = df["yamazaki_T"]
    df["delta_T"] =  df["hist_temperature"] - df["ref_T"]

    # Garder seulement observations avec delta_T valide
    df = df[df["delta_T"].notna()]

    return df

# =====================================================================
# CARTE 1 : LOCALISATION (observations AVEC delta_T)
# =====================================================================

def plot_historical_locations(df, out_path, season, depth_str, period_str):
    """Carte localisation - SEULEMENT observations AVEC delta_T valide."""
    df_valid = compute_delta_T(df)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    create_polar_map(ax, lat_limit=-60)

    ax.scatter(
        df_valid["hist_lon"],
        df_valid["hist_lat"],
        s=8,
        c="red",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    n_obs = len(df_valid)

    title = f"Localisation des observations historiques ({period_str})\n"
    title += f"Saison: {season} | Profondeur: {depth_str} | n = {n_obs:,} observations"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# =====================================================================
# CARTE 2 : GRIDDED 2°×2° (avec NoData)
# =====================================================================

def compute_grid_mean(df, grid_size=2.0):
    """Calcule moyenne ΔT par boîte grid_size×grid_size."""
    df = compute_delta_T(df)

    lat_bins = np.arange(-90, -59, grid_size)
    lon_bins = np.arange(-180, 181, grid_size)

    n_lat = len(lat_bins) - 1
    n_lon = len(lon_bins) - 1

    mean_grid = np.full((n_lat, n_lon), np.nan)

    for i in range(n_lat):
        for j in range(n_lon):
            mask = (
                (df["hist_lat"] >= lat_bins[i])
                & (df["hist_lat"] < lat_bins[i + 1])
                & (df["hist_lon"] >= lon_bins[j])
                & (df["hist_lon"] < lon_bins[j + 1])
            )

            cell_data = df[mask]["delta_T"].values

            if len(cell_data) > 0:
                mean_grid[i, j] = np.mean(cell_data)

    return mean_grid, lat_bins, lon_bins


def plot_gridded_mean_2deg(df, out_path, season, depth_str, period_str):
    """Carte gridded 2°×2° moyenne ΔT (échelle fixe -3.5 à +3.5)."""
    mean_grid, lat_bins, lon_bins = compute_grid_mean(df, grid_size=2.0)

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    create_polar_map(ax, lat_limit=-60)

    vmin, vmax = -3.5, 3.5
    LON, LAT = np.meshgrid(lon_bins, lat_bins)

    mesh = ax.pcolormesh(
        LON,
        LAT,
        mean_grid,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        zorder=2,
    )

    cbar = plt.colorbar(
        mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7
    )
    cbar.set_label("Anomalie de température (°C)", fontsize=12, fontweight="bold")

    n_valid = np.sum(~np.isnan(mean_grid))
    title = f"Anomalie de température (2°×2°) – {period_str}\n"
    title += f"Saison: {season} | Profondeur: {depth_str}\n"
    title += f"Cellules valides: {n_valid}"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================================
# PANNEAUX DÉCENNAUX 2°×2°
# =====================================================================

def plot_gridded_2deg_on_axis(ax, df, season, depth_min, depth_max, depth_str,
                               year_start, year_end, period_label, vmin=-3.5, vmax=3.5):
    """Plot une carte gridded 2°×2° sur un axe existant (pour panneaux)."""
    df_filtered = filter_data(df, season, depth_min, depth_max, year_start, year_end)
    
    if len(df_filtered) == 0:
        ax.text(0.5, 0.5, "Pas de\ndonnées", 
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    df_valid = compute_delta_T(df_filtered)
    
    if len(df_valid) < 10:
        ax.text(0.5, 0.5, "Pas assez\nd'observations",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    mean_grid, lat_bins, lon_bins = compute_grid_mean(df_filtered, grid_size=2.0)
    create_polar_map(ax, lat_limit=-60)

    LON, LAT = np.meshgrid(lon_bins, lat_bins)
    mesh = ax.pcolormesh(
        LON, LAT, mean_grid,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax,
        shading="flat", zorder=1
    )

    return mesh


def create_decadal_panels(df_all, out_dir, season):
    """Génère 3 panneaux décennaux 2°×2°."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vmin, vmax = -3.5, 3.5

    # Trouver les 2 premières profondeurs
    depth_10_100 = None
    depth_100_200 = None
    for d_min, d_max, d_str in DEPTH_RANGES:
        if "10-100m" in d_str:
            depth_10_100 = (d_min, d_max, d_str)
        elif "100-200m" in d_str:
            depth_100_200 = (d_min, d_max, d_str)

    if depth_10_100 is None or depth_100_200 is None:
        print(" Impossible de trouver les profondeurs 10-100m et 100-200m.")
        return

    d1_min, d1_max, d1_str = depth_10_100
    d2_min, d2_max, d2_str = depth_100_200

    panel1_periods = [
        (1900, 1909, "1900-1909"),
        (1910, 1919, "1910-1919"),
        (1920, 1929, "1920-1929"),
    ]

    panel2_periods = [
        (1930, 1939, "1930-1939"),
        (1940, 1949, "1940-1949"),
        (1950, 1959, "1950-1959"),
        (1960, 1969, "1960-1969"),
    ]

    panel3_periods = [
        (1900, 1929, "1900-1929"),
        (1930, 1969, "1930-1969"),
    ]

    # ---------- Panel 1 : 3 périodes (3×2)
    print("\n Génération Panel 1 (1900-1929)...")
    fig1, axes1 = plt.subplots(
        nrows=3,
        ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 12),
    )
    axes1 = np.atleast_2d(axes1)

    meshes = []
    for row, (y0, y1, label) in enumerate(panel1_periods):
        m1 = plot_gridded_2deg_on_axis(
            axes1[row, 0], df_all, season,
            d1_min, d1_max, d1_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        m2 = plot_gridded_2deg_on_axis(
            axes1[row, 1], df_all, season,
            d2_min, d2_max, d2_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        if m1 is not None:
            meshes.append(m1)
        if m2 is not None:
            meshes.append(m2)

        axes1[row, 0].text(
            -0.10, 0.5, label,
            transform=axes1[row, 0].transAxes,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
        )

    axes1[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes1[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    if meshes:
        sm1 = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="RdBu_r")
        sm1.set_array([])
        cax1 = fig1.add_axes([0.2, 0.10, 0.6, 0.025])
        cbar1 = fig1.colorbar(sm1, cax=cax1, orientation="horizontal")
        cbar1.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg1 = fig1.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg1.axis("off")
    rect1 = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0", edgecolor="black", linewidth=0.5)
    ax_leg1.add_patch(rect1)
    ax_leg1.text(1.05, 0.5, "No Data", va="center", ha="left", fontsize=9)

    fig1.suptitle(
        f"Anomalie de température (2°×2°) – {season}",
        fontsize=13, fontweight="bold", y=0.97,
    )

    fig1.subplots_adjust(
        left=0.16, right=0.95, top=0.9, bottom=0.14,
        hspace=0.15, wspace=0.05,
    )

    panel1_path = out_dir / f"YAMA_panel_gridded_2deg_{season}_1900-1929_3x2.png"
    fig1.savefig(panel1_path, dpi=300)
    plt.close(fig1)
    print(f" {panel1_path.name}")

    # ---------- Panel 2 : 4 périodes (4×2)
    print("\n Génération Panel 2 (1930-1969)...")
    fig2, axes2 = plt.subplots(
        nrows=4,
        ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 14),
    )
    axes2 = np.atleast_2d(axes2)

    meshes = []
    for row, (y0, y1, label) in enumerate(panel2_periods):
        m1 = plot_gridded_2deg_on_axis(
            axes2[row, 0], df_all, season,
            d1_min, d1_max, d1_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        m2 = plot_gridded_2deg_on_axis(
            axes2[row, 1], df_all, season,
            d2_min, d2_max, d2_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        if m1 is not None:
            meshes.append(m1)
        if m2 is not None:
            meshes.append(m2)

        axes2[row, 0].text(
            -0.10, 0.5, label,
            transform=axes2[row, 0].transAxes,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
        )

    axes2[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes2[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    if meshes:
        sm2 = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="RdBu_r")
        sm2.set_array([])
        cax2 = fig2.add_axes([0.2, 0.10, 0.6, 0.025])
        cbar2 = fig2.colorbar(sm2, cax=cax2, orientation="horizontal")
        cbar2.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg2 = fig2.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg2.axis("off")
    rect2 = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0", edgecolor="black", linewidth=0.5)
    ax_leg2.add_patch(rect2)
    ax_leg2.text(1.05, 0.5, "No Data", va="center", ha="left", fontsize=9)

    fig2.suptitle(
        f"Anomalie de température (2°×2°) – {season}",
        fontsize=13, fontweight="bold", y=0.97,
    )

    fig2.subplots_adjust(
        left=0.16, right=0.95, top=0.9, bottom=0.14,
        hspace=0.15, wspace=0.05,
    )

    panel2_path = out_dir / f"YAMA_panel_gridded_2deg_{season}_1930-1969_4x2.png"
    fig2.savefig(panel2_path, dpi=300)
    plt.close(fig2)
    print(f" {panel2_path.name}")

    # ---------- Panel 3 : 2 grandes périodes (2×2)
    print("\n Génération Panel 3 (1900-1929 vs 1930-1969)...")
    fig3, axes3 = plt.subplots(
        nrows=2,
        ncols=2,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
        figsize=(10, 10),
    )
    axes3 = np.atleast_2d(axes3)

    meshes = []
    for row, (y0, y1, label) in enumerate(panel3_periods):
        m1 = plot_gridded_2deg_on_axis(
            axes3[row, 0], df_all, season,
            d1_min, d1_max, d1_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        m2 = plot_gridded_2deg_on_axis(
            axes3[row, 1], df_all, season,
            d2_min, d2_max, d2_str,
            y0, y1, label, vmin=vmin, vmax=vmax
        )
        if m1 is not None:
            meshes.append(m1)
        if m2 is not None:
            meshes.append(m2)

        axes3[row, 0].text(
            -0.10, 0.5, label,
            transform=axes3[row, 0].transAxes,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
        )

    axes3[0, 0].set_title(d1_str, fontsize=11, fontweight="bold", pad=10)
    axes3[0, 1].set_title(d2_str, fontsize=11, fontweight="bold", pad=10)

    if meshes:
        sm3 = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="RdBu_r")
        sm3.set_array([])
        cax3 = fig3.add_axes([0.2, 0.10, 0.6, 0.025])
        cbar3 = fig3.colorbar(sm3, cax=cax3, orientation="horizontal")
        cbar3.set_label("Anomalie de température (°C)", fontsize=11)

    ax_leg3 = fig3.add_axes([0.05, 0.02, 0.12, 0.035])
    ax_leg3.axis("off")
    rect3 = Rectangle((0, 0), 1, 1, facecolor="#E0E0E0", edgecolor="black", linewidth=0.5)
    ax_leg3.add_patch(rect3)
    ax_leg3.text(1.05, 0.5, "No Data", va="center", ha="left", fontsize=9)



    fig3.subplots_adjust(
        left=0.16, right=0.95, top=0.95, bottom=0.14,
        hspace=0.15, wspace=0.05,
    )

    panel3_path = out_dir / f"YAMA_panel_gridded_2deg_{season}_1900-1969_2x2.png"
    fig3.savefig(panel3_path, dpi=300)
    plt.close(fig3)
    print(f" {panel3_path.name}")


# =====================================================================
# GÉNÉRATION DE TOUTES LES CARTES
# =====================================================================

def generate_all_maps(df_all, out_dir, season):
    """Génère toutes les cartes pour toutes les combinaisons période/profondeur."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"GÉNÉRATION DES CARTES")
    print(f"{'='*80}\n")

    for year_start, year_end, period_str in PERIODS:
        for depth_min, depth_max, depth_str in DEPTH_RANGES:
            df_filtered = filter_data(
                df_all, season, depth_min, depth_max, year_start, year_end
            )

            if len(df_filtered) == 0:
                print(f"  Skip {period_str} | {depth_str} (aucune donnée)")
                continue

            df_check = compute_delta_T(df_filtered)
            n_obs_valide = len(df_check)

            print(f"\n {period_str} | {depth_str}")
            print(f"   {len(df_filtered):,} obs totales, {n_obs_valide:,} avec delta_T valide")

            # Carte localisation
            loc_path = out_dir / f"YAMA_locations_{season}_{depth_str}_{period_str}.png"
            plot_historical_locations(df_filtered, loc_path, season, depth_str, period_str)
            print(f"    Localisation : {loc_path.name}")

            # Carte gridded 2°×2°
            grid2_path = out_dir / f"YAMA_gridded_2deg_{season}_{depth_str}_{period_str}.png"
            plot_gridded_mean_2deg(df_filtered, grid2_path, season, depth_str, period_str)
            print(f"    Griddée 2°×2° : {grid2_path.name}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Générateur cartes polaires Yamazaki vs Historiques")
    ap.add_argument("--csv-dir", type=str, required=True,
                    help="Répertoire des CSV (ex: ~/Thesis/Yamazaki/Datas)")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Répertoire de sortie (ex: ~/Thesis/Yamazaki/Outputs)")
    ap.add_argument(
        "--season",
        type=str,
        default="DJF",
        choices=["DJF", "MAM", "JJA", "SON", "ALL"],
        help="Saison à traiter"
    )

    args = ap.parse_args()

    print("=" * 80)
    print("GÉNÉRATEUR CARTES POLAIRES - Yamazaki vs Historiques")
    print("=" * 80)
    print(f"Saison: {args.season}")
    print(f"Échelle: -3.5 à +3.5°C (FIXE)")
    print()

    df_all = load_csv_data(Path(args.csv_dir))
    generate_all_maps(df_all, args.out_dir, args.season)

    # Panneaux 2°×2° décennaux
    create_decadal_panels(df_all, args.out_dir, args.season)

    print(f"\n{'='*80}")
    print(" TERMINÉ")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()