"""
Commande d'execution : python3 ~/Thesis/WOD/Codes/wod_polar_coverage_2deg_OK.py

WOD → Cartes polaires Antarctique (2°×2°)
- 7 panneaux (1900–1909 … 1960–1969)
- TOTAL et MOYENNE/AN stricte (= somme annuelle / 10)
- Profondeurs : 10–100, 100–200 m
- Palette/normalisation/projection identiques à tes versions validées
- I/O adaptés à la structure: Codes/ Dataset/ Outputs
- NOUVEAU: échelle de légende paramétrable :
    * 'wide'  (défaut) → breaks = [0,20,40,100,1000,2000]  (étiquette finale '>2000')
    * '50'             → breaks = [0,5,10,20,40,50]        (étiquette finale  '>50')
  Sélection via variable d'environnement WOD_SCALE = wide | 50
"""

import os, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import num2date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPat
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from pathlib import Path

# ---------- Configuration globale des fontes ----------
FONT_BASE          = 11  # base générale
FONT_PANEL_TITLE   = 13  # décennie au-dessus de chaque carte
FONT_SUPTITLE      = 14  # titre global de la figure 
FONT_CBAR_LABEL    = 13  # label de la barre de couleur
FONT_CBAR_TICKS    = 13  # valeurs sur la barre de couleur
FONT_AX_TICKS      = 10  # taille potentielle des labels d'axes

mpl.rcParams.update({
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_PANEL_TITLE,
    "axes.labelsize": FONT_BASE,
    "xtick.labelsize": FONT_AX_TICKS,
    "ytick.labelsize": FONT_AX_TICKS,
    "figure.titlesize": FONT_SUPTITLE,
    "legend.fontsize": FONT_BASE,
})

# ---------- Dossiers (Codes/../Dataset et Outputs) ----------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../WOD/Codes
BASE_DIR   = SCRIPT_DIR.parent                         # .../WOD
DATA_DIR   = Path(os.environ.get("WOD_DATA_DIR", str(BASE_DIR / "Datas"))).resolve()
OUT_DIR    = Path(os.environ.get("WOD_OUT_DIR",  str(BASE_DIR / "Outputs/Outputs_OK"))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECURSIVE  = bool(int(os.environ.get("WOD_RECURSIVE", "0")))
LAT_CUTOFF = -60
ENGINE     = os.environ.get("WOD_ENGINE", "scipy")     

# ---------- Échelles de légende ----------
def get_breaks_and_label(scale_type):
    """Retourne (BREAKS numpy array, last_label) selon le type d'échelle."""
    if scale_type == "mean":
        # échelle courte pour moyennes par an : 0..>50
        br = np.array([0, 5, 10, 20, 40, 50], dtype=float)
        return br, ">50"
    else:  # scale_type == "total"
        # échelle large pour totaux : 0..>2000
        br = np.array([0, 20, 40, 100, 1000, 2000], dtype=float)
        return br, ">2000"

# ---------- Noms des fichiers de sortie (dans OUT_DIR) ----------
OUT_FIG_TOTAL_2DEG = OUT_DIR / f"wod_coverage_observations_2deg.png"
OUT_FIG_MEAN_2DEG  = OUT_DIR / f"wod_coverage_observations_mean_per_year_2deg.png"

# Profondeurs
DEPTH_BANDS = [(10,100), (100,200)]
def depth_tag(lo, hi): return f"{lo:03d}_{hi:03d}"

# Tranches 10 ans (7 panneaux)
BINS = [(1900,1909),(1910,1919),(1920,1929),(1930,1939),(1940,1949),(1950,1959),(1960,1969)]

# Grille 2°×2° (centres)
LATS2 = np.arange(-89.0,  90.0,  2.0)
LONS2 = np.arange(-179.0, 180.0, 2.0)

# ---------- Palette & normalisation (paramétrées dynamiquement) ----------
def make_norm_and_cmap(breaks):
    """Crée la normalisation et la colormap pour un ensemble de breaks donné."""
    _B = breaks.astype(float)
    _POS = np.linspace(0.0, 1.0, _B.size)

    def _forward(values):
        v = np.asarray(values, dtype=float)
        v = np.clip(v, _B[0], _B[-1])
        return np.interp(v, _B, _POS)

    def _inverse(frac):
        f = np.asarray(frac, dtype=float)
        f = np.clip(f, 0.0, 1.0)
        return np.interp(f, _POS, _B)

    try:
        norm = mpl.colors.FuncNorm((_forward, _inverse), vmin=_B[0], vmax=_B[-1])
    except Exception:
        norm = mpl.colors.Normalize(vmin=_B[0], vmax=_B[-1])

    # Couleurs ancrées exactement aux valeurs demandées + stops internes jaune/orange
    c0     = "#FFFFFF"  # 0
    c20    = "#84FFFF"  # 20
    c40    = "#4E57FB"  # 40
    c100   = "#275C57"  # 100
    c1000  = "#8AFF6F"  # 1000
    c1500  = "#FFEF04"  # ~1500 (jaune)
    c1800  = "#FE8E07"  # ~1800 (orange)
    c2000  = "#FB0005"  # 2000

    p0, p20, p40, p100, p1000, p2000 = _POS
    p1500 = p1000 + (p2000 - p1000) * (1500 - 1000) / (2000 - 1000)
    p1800 = p1000 + (p2000 - p1000) * (1800 - 1000) / (2000 - 1000)

    CMAP_STOPS = [
        (p0,    c0),
        (p20,   c20),
        (p40,   c40),
        (p100,  c100),
        (p1000, c1000),
        (p1500, c1500),
        (p1800, c1800),
        (p2000, c2000),
    ]
    cmap = LinearSegmentedColormap.from_list("antarctica_custom", CMAP_STOPS, N=512)
    return norm, cmap

# ---------- Utilitaires visuels ----------
def circle_boundary_path():
    th = np.linspace(0, 2*np.pi, 256)
    v  = np.vstack([0.5+0.5*np.cos(th), 0.5+0.5*np.sin(th)]).T
    return MplPath(v)

def setup_ax(ax):
    ax.set_extent([-180, 180, -90, LAT_CUTOFF], ccrs.PlateCarree())
    ax.set_boundary(circle_boundary_path(), transform=ax.transAxes)
    ax.set_facecolor("#E0E0E0")  # gris clair
    ax.add_feature(cfeature.LAND, facecolor="antiquewhite", edgecolor="none", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, color="black", zorder=4)
    ylocs = np.arange(-90, int(LAT_CUTOFF)-9, 10)
    try:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     xlocs=np.arange(-180,181,30), ylocs=ylocs,
                     linestyle="--", linewidth=0.6, color="0.5", zorder=5)
    except TypeError:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                     xlocs=np.arange(-180,181,30), ylocs=ylocs,
                     linestyle="--", linewidth=0.6, color="0.5")
    
    ax.tick_params(labelsize=FONT_AX_TICKS)

def plot_panels(OUT_PATH, panels_arrays, title, cbar_label, lat_centers, lon_centers, scale_type="total"):
    """
    scale_type: "total" pour échelle large [0,20,40,100,1000,>2000]
                "mean" pour échelle courte [0,5,10,20,40,>50]
    """
    BREAKS, LAST_LAB = get_breaks_and_label(scale_type)
    NORM_CONT, CMAP_CONT = make_norm_and_cmap(BREAKS)
    
    proj = ccrs.SouthPolarStereo()
    fig  = plt.figure(figsize=(16, 8))
    gs   = mpl.gridspec.GridSpec(2, 4, figure=fig, hspace=0.12, wspace=0.04)
    axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(4)]
    axes+= [fig.add_subplot(gs[1, i], projection=proj) for i in range(3)]

    dlat = float(lat_centers[1]-lat_centers[0]); dlon = float(lon_centers[1]-lon_centers[0])
    lat_edges = np.r_[lat_centers-dlat/2.0, lat_centers[-1]+dlat/2.0]
    lon_edges = np.r_[lon_centers-dlon/2.0, lon_centers[-1]+dlon/2.0]

    vmax = BREAKS[-1]
    for ax, (A, (y0, y1)) in zip(axes, panels_arrays):
        setup_ax(ax)
        if A is None:
            ax.set_title(f"{y0}-{y1} (aucune donnée)",
                         fontsize=FONT_PANEL_TITLE, pad=6)
            continue
        Aplot = np.minimum(A.astype(float), vmax)
        ax.pcolormesh(lon_edges, lat_edges, Aplot, transform=ccrs.PlateCarree(),
                      cmap=CMAP_CONT, norm=NORM_CONT, shading="auto", zorder=1)
        # Titre de chaque sous-figure = décennie
        ax.set_title(f"{y0}-{y1}", fontsize=FONT_PANEL_TITLE, pad=6)

    # Titre global : seulement si title non None
    if title is not None:
        fig.suptitle(title, fontsize=FONT_SUPTITLE, y=0.98)

    # Colorbar
    cax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
    cb  = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM_CONT, cmap=CMAP_CONT),
                       cax=cax, orientation="horizontal", ticks=BREAKS)
    ticks = [str(int(v)) if v < BREAKS[-1] else LAST_LAB for v in BREAKS]
    cb.set_ticklabels(ticks)
    cb.ax.tick_params(labelsize=FONT_CBAR_TICKS)
    cb.set_label(cbar_label, fontsize=FONT_CBAR_LABEL)

    plt.subplots_adjust(top=0.92, bottom=0.11, left=0.04, right=0.98)
    fig.savefig(str(OUT_PATH), dpi=220, bbox_inches="tight")
    print("Écrit :", OUT_PATH)

# ---------- Lecture WOD (ragged arrays) ----------
def is_sst_file_wod(path):
    n = os.path.basename(path).upper()
    return ("_SUR" in n) or ("_DRB" in n) or n.startswith("SUR_") or n.startswith("DRB_")

def iter_wod_files(root: Path, recursive=True):
    if recursive:
        yield from (p for p in root.rglob("*.nc") if p.is_file())
    else:
        yield from (p for p in root.glob("*.nc") if p.is_file())

def _safe_open(path: Path):
    for eng in [ENGINE, "scipy", "netcdf4"]:
        try:
            return xr.open_dataset(path, engine=eng, decode_times=False)
        except Exception:
            pass
    return None

def _centers_to_indices(lat_vals, lon_vals, lat_centers, lon_centers):
    dlat=float(lat_centers[1]-lat_centers[0]); dlon=float(lon_centers[1]-lon_centers[0])
    lat0=lat_centers[0]-dlat/2.0; lon0=lon_centers[0]-dlon/2.0
    i=np.floor((lat_vals-lat0)/dlat).astype(int); j=np.floor((lon_vals-lon0)/dlon).astype(int)
    i=np.clip(i,0,lat_centers.size-1); j=np.clip(j,0,lon_centers.size-1); return i,j

def year_from_time_var(time_arr, units, calendar="standard"):
    try:
        dt = num2date(time_arr, units=units, calendar=calendar)
        return np.array([d.year for d in np.atleast_1d(dt)], dtype=int)
    except Exception:
        return np.full(time_arr.shape, -999, dtype=int)

# ---------- Accumulation ----------
def accumulate_counts_wod(data_dir: Path, recursive, lat_centers, lon_centers, depth_bands, bins_10y):
    totals = {tuple(b): np.zeros((lat_centers.size, lon_centers.size), dtype=np.int64) for b in bins_10y}
    annual = {tuple(b): defaultdict(lambda: np.zeros((lat_centers.size, lon_centers.size), dtype=np.int64))
              for b in bins_10y}

    for path in iter_wod_files(data_dir, recursive):
        ds = _safe_open(path)
        if ds is None: continue
        if not all(k in ds for k in ("lat","lon","time")): ds.close(); continue

        latc=np.asarray(ds["lat"].values); lonc=np.asarray(ds["lon"].values)
        timev=np.asarray(ds["time"].values)
        t_units=ds["time"].attrs.get("units","days since 1770-01-01 00:00:00 UTC")
        t_cal  =ds["time"].attrs.get("calendar","standard")
        years  =year_from_time_var(timev, t_units, t_cal)

        lonc=((lonc+180.0)%360.0)-180.0
        latc=np.clip(latc,-89.999,89.999); lonc=np.clip(lonc,-179.999,179.999)

        has_temp=("Temperature" in ds) and ("Temperature_row_size" in ds) and ("Temperature_WODflag" in ds)
        has_z   =("z" in ds) and ("z_row_size" in ds)
        if not (has_temp and has_z): ds.close(); continue

        temp = np.asarray(ds["Temperature"].values)
        tflag= np.asarray(ds["Temperature_WODflag"].values)
        z    = np.asarray(ds["z"].values)
        row_t= np.asarray(ds["Temperature_row_size"].values).astype(int)
        row_z= np.asarray(ds["z_row_size"].values).astype(int)
        ofs_t= np.r_[0, np.cumsum(row_t)[:-1]]
        ofs_z= np.r_[0, np.cumsum(row_z)[:-1]]

        is_sst = is_sst_file_wod(str(path))

        for ic in range(latc.shape[0]):
            y=int(years[ic]); la=float(latc[ic]); lo=float(lonc[ic])
            if la > -60.0:
                continue

            t0=int(ofs_t[ic]); tN=int(ofs_t[ic]+row_t[ic])
            z0=int(ofs_z[ic]); zN=int(ofs_z[ic]+row_z[ic])
            t0=max(0,min(t0,temp.size)); tN=max(0,min(tN,temp.size))
            z0=max(0,min(z0,z.size));    zN=max(0,min(zN,z.size))
            if tN<=t0 or zN<=z0: continue

            mqc = (tflag[t0:tN] == 0)

            if is_sst:
                n_obs=int(np.count_nonzero(mqc))
                if n_obs==0: continue
                ii,jj=_centers_to_indices(np.array([la]), np.array([lo]), lat_centers, lon_centers)
                for (y0,y1) in bins_10y:
                    if y0<=y<=y1:
                        totals[(y0,y1)][ii[0],jj[0]]+=n_obs
                        annual[(y0,y1)][y][ii[0],jj[0]]+=n_obs
            else:
                n_pair=min(tN-t0, zN-z0)
                if n_pair<=0: continue
                z_slice=z[z0:z0+n_pair]; mqc_slice= mqc[:n_pair]
                latv=np.full(n_pair, la); lonv=np.full(n_pair, lo)
                ii,jj=_centers_to_indices(latv, lonv, lat_centers, lon_centers)
                for (lo_d,hi_d) in depth_bands:
                    mband = mqc_slice & np.isfinite(z_slice) & (z_slice>=lo_d) & (z_slice<hi_d)
                    if not np.any(mband): continue
                    for (y0,y1) in bins_10y:
                        if y0<=y<=y1:
                            np.add.at(totals[(y0,y1)], (ii[mband],jj[mband]), 1)
                            np.add.at(annual[(y0,y1)][y], (ii[mband],jj[mband]), 1)
        ds.close()

    panels_total, panels_mean = [], []
    for (y0,y1) in bins_10y:
        A_tot = totals[(y0,y1)]
        if len(annual[(y0,y1)])==0:
            A_mean=None
        else:
            S=np.zeros_like(A_tot, dtype=float)
            for yy in range(y0,y1+1):
                S += annual[(y0,y1)][yy].astype(float)
            A_mean = S/10.0
        panels_total.append((A_tot if np.any(A_tot) else None, (y0,y1)))
        panels_mean.append((A_mean,(y0,y1)))
    return panels_total, panels_mean

# ---------- Orchestrations ----------
def build_and_plot_totals_and_means_2deg_wod():
    panels_total, panels_mean = accumulate_counts_wod(DATA_DIR, RECURSIVE, LATS2, LONS2,
                                                      depth_bands=[(0,1e9)], bins_10y=BINS)
    # Figures globales : pas de titre (évite la redondance avec la légende)
    plot_panels(OUT_FIG_TOTAL_2DEG, panels_total,
                None,
                cbar_label="Nombre d'observations (2°×2°)",
                lat_centers=LATS2, lon_centers=LONS2, scale_type="total")
    plot_panels(OUT_FIG_MEAN_2DEG, panels_mean,
                None,
                cbar_label="Nombre moyen d'observations par an (2°×2°)",
                lat_centers=LATS2, lon_centers=LONS2, scale_type="mean")

def build_and_plot_depth_total_and_means_2deg_wod():
    for (lo_d,hi_d) in DEPTH_BANDS:
        panels_total, panels_mean = accumulate_counts_wod(DATA_DIR, RECURSIVE, LATS2, LONS2,
                                                          depth_bands=[(lo_d,hi_d)], bins_10y=BINS)
        out_total = OUT_DIR / f"wod_coverage_observations_2deg_depth_{depth_tag(lo_d,hi_d)}.png"
        out_mean  = OUT_DIR / f"wod_coverage_observations_mean_per_year_2deg_depth_{depth_tag(lo_d,hi_d)}.png"
        plot_panels(out_total, panels_total,
                    title=f"Profondeur {lo_d}-{hi_d} m",
                    cbar_label="Nombre d'observations (2°×2°)",
                    lat_centers=LATS2, lon_centers=LONS2, scale_type="total")
        plot_panels(out_mean, panels_mean,
                    title=f"Profondeur {lo_d}-{hi_d} m",
                    cbar_label="Nombre moyen d'observations par an (2°×2°)",
                    lat_centers=LATS2, lon_centers=LONS2, scale_type="mean")

# ---------- Main ----------
if __name__ == "__main__":
    print(f"[WOD] DATA_DIR={DATA_DIR}")
    print(f"[WOD] OUT_DIR ={OUT_DIR}")
    print(f"[WOD] RECURSIVE={RECURSIVE} ENGINE={ENGINE}")
    build_and_plot_totals_and_means_2deg_wod()
    build_and_plot_depth_total_and_means_2deg_wod()
