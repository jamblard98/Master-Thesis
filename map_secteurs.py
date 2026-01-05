"""
map_sea_locations_v4.py
Commande d'éxécution :
python3 map_secteurs.py \
  --out-dir ~/Thesis/Outputs/figures \
  --background-tif ~/Thesis/Antarctic_Imagery_3031.tif
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
import shapely.geometry as sgeom

# Optionnel : rasterio pour lire un GeoTIFF en fond
try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

# =====================================================================
# CONFIGURATION
# =====================================================================

# Transparence des boîtes (0 = opaque, 1 = invisible)
BOX_ALPHA = 0.30  # (0.2–0.4)

# Longitude à laquelle placer TOUS les labels de latitude (°E ; négatif = °W)
# Ici : 50°E 
LON_LABEL_INSIDE = 50

SEAS = {
    'Ross': {
        'bounds': (-78, -70, 210, 160),  # (lat_min, lat_max, lon_min, lon_max)
        'color': '#f39c12',  # Orange
        'label': 'Ross'
    },
    'Weddell': {
        'bounds': (-78, -60, -60, -20),
        'color': '#ffff00',  # Jaune
        'label': 'Weddell'
    },
    'BA': {
        'bounds': (-75, -65, -130, -60),  # Bellingshausen/Amundsen
        'color': '#e74c3c',  # Rouge
        'label': 'Bellingshausen/Amundsen'
    },
    'Davis': {
        'bounds': (-68, -65, 80, 100),
        'color': '#2ecc71',  # Vert
        'label': 'Davis'
    }
}

# =====================================================================
# FONCTIONS CARTE
# =====================================================================

def circle_boundary_path():
    """Path circulaire pour le cadre."""
    th = np.linspace(0, 2 * np.pi, 256)
    v = np.vstack([0.5 + 0.5 * np.cos(th),
                   0.5 + 0.5 * np.sin(th)]).T
    return MplPath(v)


def add_raster_background(ax, tif_path):
    """
    Ajoute un fond raster (GeoTIFF local) sur l'axe.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio n'est pas installé (conda install rasterio).")

    tif_path = Path(tif_path)
    if not tif_path.is_file():
        raise FileNotFoundError(f"Fichier raster introuvable : {tif_path}")

    with rasterio.open(tif_path) as src:
        data = src.read()  # (bands, y, x)
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)  # → (y, x, bands)
        bounds = src.bounds
        crs_str = src.crs.to_string().lower() if src.crs is not None else ""

        if "3031" in crs_str:
            img_crs = ccrs.SouthPolarStereo()
        elif "4326" in crs_str or "wgs 84" in crs_str or "wgs84" in crs_str:
            img_crs = ccrs.PlateCarree()
        else:
            img_crs = ccrs.PlateCarree()

        ax.imshow(
            data,
            origin="upper",
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
            transform=img_crs,
            zorder=1
        )


def create_polar_map(ax, lat_limit=-60, background_tif=None):
    """
    Configure projection stéréographique polaire Sud.
    """
    ax.set_extent([-180, 180, -90, lat_limit], crs=ccrs.PlateCarree())
    ax.set_boundary(circle_boundary_path(), transform=ax.transAxes)

    if background_tif is not None:
        try:
            print(f"   → Ajout du fond raster : {background_tif}")
            add_raster_background(ax, background_tif)
        except Exception as exc:
            print(f"     Impossible d'utiliser le fond raster ({exc}).")
            print("      → Utilisation d'un fond simple.")
            ax.set_facecolor("white")
            ax.add_feature(
                cfeature.LAND,
                facecolor="lightgray",
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )
    else:
        ax.set_facecolor("white")
        ax.add_feature(
            cfeature.LAND,
            facecolor="lightgray",
            edgecolor="none",
            linewidth=0.0,
            zorder=2,
        )

    # Grille lat/lon
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, lat_limit + 1, 10),
        linewidth=0.6,
        color="gray",
        alpha=0.4,
        linestyle="--",
        zorder=4,
    )

    # Labels 60, 70, 80°S tous au même méridien (50°E ici)
    # 60°S est légèrement "poussé" vers l'intérieur (−59.5°S) pour rester lisible.
    lats_to_label = [-60, -70, -80]
    for lat in lats_to_label:
        if lat == -60:
            lat_plot = lat + 0.5   # on place "60°S" à 59.5–59°S pour qu'il ne soit plus sur l'océan
        else:
            lat_plot = lat
        ax.text(
            LON_LABEL_INSIDE, lat_plot,
            f"{abs(lat)}°S",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=7,
        )

    return ax


def create_latitude_following_box(lat_min, lat_max, lon_min, lon_max):
    """Crée un polygone Shapely qui suit EXACTEMENT les latitudes."""
    if lon_max > 180:
        lon_max = lon_max - 360

    if lon_min < lon_max:
        lons_north = np.linspace(lon_min, lon_max, 100)
    else:
        lons_north = np.linspace(lon_min, lon_max, 100)
    lats_north = np.full_like(lons_north, lat_max)

    lats_east = np.linspace(lat_max, lat_min, 20)
    lons_east = np.full_like(lats_east, lon_max)

    lons_south = np.linspace(lon_max, lon_min, 100)
    lats_south = np.full_like(lons_south, lat_min)

    lats_west = np.linspace(lat_min, lat_max, 20)
    lons_west = np.full_like(lats_west, lon_min)

    lons = np.concatenate([lons_north, lons_east, lons_south, lons_west])
    lats = np.concatenate([lats_north, lats_east, lats_south, lats_west])

    coords = list(zip(lons, lats))
    polygon = sgeom.Polygon(coords)
    return polygon


def add_sea_box(ax, lat_min, lat_max, lon_min, lon_max, color, alpha=BOX_ALPHA):
    """Ajoute une boîte géodésique suivant les latitudes."""
    polygon = create_latitude_following_box(lat_min, lat_max, lon_min, lon_max)
    ax.add_geometries(
        [polygon],
        crs=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2.5,
        zorder=6,
    )


def create_sea_location_map(out_dir, background_tif=None):
    """Génère la carte de localisation des mers."""
    print("\n Génération carte de localisation des mers...")

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    # IMPORTANT : background_tif
    create_polar_map(ax, lat_limit=-60, background_tif=background_tif)

    legend_handles = []
    for sea_name, sea_info in SEAS.items():
        lat_min, lat_max, lon_min, lon_max = sea_info['bounds']
        color = sea_info['color']
        label = sea_info['label']

        add_sea_box(ax, lat_min, lat_max, lon_min, lon_max, color, alpha=BOX_ALPHA)

        legend_handles.append(
            mpatches.Rectangle(
                (0, 0), 1, 1,
                fc=color, ec=color,
                alpha=BOX_ALPHA,
                linewidth=2.5,
                label=label
            )
        )

        if lon_min < 0 and lon_max < 0:
            lon_str = f"{abs(lon_min)}°W–{abs(lon_max)}°W"
        elif lon_min >= 0 and lon_max >= 0:
            lon_str = f"{lon_min}°E–{lon_max}°E"
        elif lon_max > 180:
            lon_str = f"{lon_min}°E–{360 - lon_max}°W"
        else:
            lon_str = f"{lon_min}°–{lon_max}°"

        lat_str = f"{abs(lat_min)}°S–{abs(lat_max)}°S"
        print(f"   ✓ {sea_name:10s}: {lat_str:15s} × {lon_str}")

    sea_order = ['Ross', 'Weddell', 'BA', 'Davis']
    ordered_handles = [legend_handles[list(SEAS.keys()).index(s)] for s in sea_order]

    ax.legend(
        handles=ordered_handles,
        loc='upper right',
        bbox_to_anchor=(1.18, 1.0),
        fontsize=12,
        frameon=True,
        title='Mers',
        title_fontsize=13
    )

    out_path = out_dir / "Map_SeaLocations_SouthernOcean.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n Carte sauvegardée : {out_path.name}")
    return out_path


# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Carte de localisation avec boîtes géodésiques"
    )
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Répertoire de sortie")
    ap.add_argument("--background-tif", type=str, default=None,
                    help="Chemin vers un GeoTIFF d'imagerie (optionnel)")

    args = ap.parse_args()

    out_dir = Path(args.out-dir) if hasattr(args, "out-dir") else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background_tif = args.background_tif

    print("=" * 80)
    print("CARTE DE LOCALISATION — BOÎTES GÉODÉSIQUES SHAPELY")
    print("=" * 80)

    create_sea_location_map(out_dir, background_tif=background_tif)

    print(f"\n{'=' * 80}\n TERMINÉ\n{'=' * 80}")


if __name__ == "__main__":
    main()
