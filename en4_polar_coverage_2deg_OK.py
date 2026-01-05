"""
Commande d'exécution :
    python3 ~/Thesis/EN4/Codes/en4_polar_coverage_2deg_OK.py
EN4 → Cartes polaires Antarctique (2°×2°)
- 7 panneaux (1900–1909 … 1960–1969)
- TOTAL et MOYENNE/AN stricte (= somme annuelle / 10)
- Profondeurs : 10–100, 100–200 m
- Palette/normalisation/projection identiques à tes versions validées
"""

import os, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from pathlib import Path

# ---------- Configuration globale des fontes ----------
FONT_BASE          = 11  # base générale
FONT_PANEL_TITLE   = 13  # décennie au-dessus de chaque carte
FONT_SUPTITLE      = 14  # titre global de la figure
FONT_CBAR_LABEL    = 13  # label de la barre de couleur
FONT_CBAR_TICKS    = 13  # valeurs sur la barre de couleur
FONT_AX_TICKS      = 10  # si des ticks axes X/Y sont visibles un jour

mpl.rcParams.update({
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_PANEL_TITLE,
    "axes.labelsize": FONT_BASE,
    "xtick.labelsize": FONT_AX_TICKS,
    "ytick.labelsize": FONT_AX_TICKS,
    "figure.titlesize": FONT_SUPTITLE,
    "legend.fontsize": FONT_BASE,
})

# ---------- Dossiers ----------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
DATA_DIR   = Path(os.environ.get("EN4_DATA_DIR", str(BASE_DIR / "Datas"))).resolve()
OUT_DIR    = Path(os.environ.get("EN4_OUT_DIR",  str(BASE_DIR / "Outputs/Outputs_OK"))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECURSIVE  = bool(int(os.environ.get("EN4_RECURSIVE", "1")))
LAT_CUTOFF = -60

# ---------- Échelles de légende ----------
def get_breaks_and_label(scale_type):
    """Retourne (BREAKS numpy array, last_label) selon le type d'échelle."""
    if scale_type == "mean":
        br = np.array([0, 5, 10, 20, 40, 50], dtype=float)
        return br, ">50"
    else:  # scale_type == "total"
        br = np.array([0, 20, 40, 100, 1000, 2000], dtype=float)
        return br, ">2000"

# ---------- Noms des fichiers de sortie ----------
OUT_FIG_TOTAL_2DEG = OUT_DIR / "en4_coverage_observations_2deg.png"
OUT_FIG_MEAN_2DEG  = OUT_DIR / "en4_coverage_observations_mean_per_year_2deg.png"

# Profondeurs
DEPTH_BANDS = [(10,100), (100,200)]
def depth_tag(lo, hi): return f"{lo:03d}_{hi:03d}"

# Tranches 10 ans (7 panneaux)
BINS = [(1900,1909),(1910,1919),(1920,1929),(1930,1939),(1940,1949),(1950,1959),(1960,1969)]

# Grille 2°×2° (centres)
LATS2 = np.arange(-89.0,  90.0,  2.0)
LONS2 = np.arange(-179.0, 180.0, 2.0)

# ---------- Palette & normalisation ----------
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

    c0="#FFFFFF"; c20="#84FFFF"; c40="#4E57FB"; c100="#275C57"; c1000="#8AFF6F"
    c1500="#FFEF04"; c1800="#FE8E07"; c2000="#FB0005"

    p0, p20, p40, p100, p1000, p2000 = _POS
    p1500 = p1000 + (p2000 - p1000) * (1500 - 1000) / (2000 - 1000)
    p1800 = p1000 + (p2000 - p1000) * (1800 - 1000) / (2000 - 1000)

    CMAP_STOPS = [
        (p0, c0), (p20, c20), (p40, c40), (p100, c100),
        (p1000, c1000), (p1500, c1500), (p1800, c1800), (p2000, c2000),
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
    ax.add_feature(cfeature.LAND, facecolor="antiquewhite", edgecolor="none", zorder=3)  # antiquewhite
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, color="black", zorder=4)  #  50m 
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

    # Titre global
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

# ---------- Lecture EN4 (profils 2D) ----------
def iter_en4_files(root: Path, recursive=True):
    if recursive:
        paths = list(root.rglob("*.nc"))
    else:
        paths = list(root.glob("*.nc"))
    # Garder seulement les fichiers .f.profiles (finalisés)
    return [p for p in paths if ".f.profiles." in p.name]

def _safe_open(path: Path):
    for eng in ("netcdf4", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng, decode_times=False)
        except Exception:
            pass
    return None

def pick(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    return None

def parse_days_since(units):
    if not isinstance(units, str) or "since" not in units.lower():
        base = datetime(1600, 1, 1)
        scale = "days"
    else:
        u = units.strip()
        scale = u.split()[0].lower()
        ref = u.lower().split("since", 1)[1].strip()
        ref_date = ref.split()[0]
        y, m, d = (int(ref_date[0:4]), int(ref_date[5:7]), int(ref_date[8:10]))
        base = datetime(y, m, d)
    fac = {"seconds":1, "minutes":60, "hours":3600, "days":86400}.get(scale, 86400)
    def conv(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return base + timedelta(seconds=float(x) * fac)
        except Exception:
            return None
    return conv

def years_from_time_var(time_values, units):
    conv = parse_days_since(units)
    flat = np.asarray(time_values).ravel()
    out = np.empty(flat.size, dtype=np.int32)
    for i, v in enumerate(flat):
        dt = conv(v)
        out[i] = dt.year if dt is not None else -9999
    return out

def _centers_to_indices(lat_vals, lon_vals, lat_centers, lon_centers):
    dlat=float(lat_centers[1]-lat_centers[0]); dlon=float(lon_centers[1]-lon_centers[0])
    lat0=lat_centers[0]-dlat/2.0; lon0=lon_centers[0]-dlon/2.0
    i=np.floor((lat_vals-lat0)/dlat).astype(int); j=np.floor((lon_vals-lon0)/dlon).astype(int)
    i=np.clip(i,0,lat_centers.size-1); j=np.clip(j,0,lon_centers.size-1)
    return i,j

# ---------- Accumulation ----------
def accumulate_counts_en4(data_dir: Path, recursive, lat_centers, lon_centers, depth_bands, bins_10y):
    totals = {tuple(b): np.zeros((lat_centers.size, lon_centers.size), dtype=np.int64) for b in bins_10y}
    annual = {tuple(b): defaultdict(lambda: np.zeros((lat_centers.size, lon_centers.size), dtype=np.int64))
              for b in bins_10y}

    # Variables candidates EN4
    LAT = ["LATITUDE","latitude","LAT","lat"]
    LON = ["LONGITUDE","longitude","LON","lon"]
    TIME = ["JULD","TIME","time","juld"]
    DEPH = ["DEPH_CORRECTED","DEPH","DEPTH","Z"]
    PRES = ["PRES","PRES_CORRECTED","PRESSURE","pressure"]
    TVAR = ["POTM_CORRECTED","TEMP","POTM","TEMPERATURE"]
    QCVAR_FOR_POTM = ["POTM_CORRECTED_QC","POTM_QC","TEMP_QC","TEMPERATURE_QC"]
    QCVAR_FOR_TEMP = ["TEMP_QC","TEMPERATURE_QC","POTM_CORRECTED_QC","POTM_QC"]
    
    accept_qc = {'1', '0', 1, 0}  # QC acceptés pour EN4

    for path in iter_en4_files(data_dir, recursive):
        ds = _safe_open(path)
        if ds is None: continue
        
        try:
            v_lat = pick(ds, LAT)
            v_lon = pick(ds, LON)
            v_time = pick(ds, TIME)
            v_deph = pick(ds, DEPH)
            v_pres = pick(ds, PRES)
            v_t = pick(ds, TVAR)

            if not all([v_lat, v_time, (v_deph or v_pres), v_t]):
                continue

            # QC associé à la température choisie
            if v_t in ("POTM_CORRECTED", "POTM"):
                v_qc = pick(ds, QCVAR_FOR_POTM)
            else:
                v_qc = pick(ds, QCVAR_FOR_TEMP)

            lat = np.asarray(ds[v_lat].values).astype(float)       # (N_PROF,)
            lon = np.asarray(ds[v_lon].values).astype(float) if v_lon else np.zeros_like(lat)
            timev = np.asarray(ds[v_time].values).astype(float)    # (N_PROF,)
            units = ds[v_time].attrs.get("units", "")
            years = years_from_time_var(timev, units)

            # profondeur 2D (N_PROF x N_LEVELS)
            if v_deph:
                z = np.asarray(ds[v_deph].values).astype(float)
            else:
                z = np.asarray(ds[v_pres].values).astype(float)

            # QC 2D si dispo
            qc_arr = None
            if v_qc and (v_qc in ds.variables):
                qc = ds[v_qc].values
                qc_arr = np.asarray(qc).astype(str) if getattr(qc, "dtype", None) and (qc.dtype.kind in "USO") else np.asarray(qc)

            # Wrap longitude
            lon = ((lon + 180.0) % 360.0) - 180.0
            lat = np.clip(lat, -89.999, 89.999)
            lon = np.clip(lon, -179.999, 179.999)

            n_prof = lat.shape[0]

            for ip in range(n_prof):
                y = int(years[ip])
                la = float(lat[ip])
                lo = float(lon[ip])
                
                # Filtres
                if la > -60.0:  # Garder seulement 60°S-90°S
                    continue
                
                # Vérifier la période
                in_bin = False
                for (y0, y1) in bins_10y:
                    if y0 <= y <= y1:
                        in_bin = True
                        break
                if not in_bin:
                    continue

                zi = z[ip, :] if z.ndim == 2 else z[:]
                if zi.size == 0:
                    continue
                    
                if qc_arr is not None:
                    qci = qc_arr[ip, :] if qc_arr.ndim == 2 else qc_arr[:]
                    ok_qc = np.array([q in accept_qc for q in qci], dtype=bool)
                else:
                    ok_qc = np.ones_like(zi, dtype=bool)

                # Boucle sur les bandes de profondeur
                for (lo_d, hi_d) in depth_bands:
                    ok = np.isfinite(zi) & ok_qc & (zi >= lo_d) & (zi < hi_d)
                    if not np.any(ok):
                        continue
                    
                    n_obs = int(ok.sum())
                    latv = np.full(n_obs, la)
                    lonv = np.full(n_obs, lo)
                    ii, jj = _centers_to_indices(latv, lonv, lat_centers, lon_centers)
                    
                    for (y0, y1) in bins_10y:
                        if y0 <= y <= y1:
                            np.add.at(totals[(y0,y1)], (ii, jj), 1)
                            np.add.at(annual[(y0,y1)][y], (ii, jj), 1)

        finally:
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
def build_and_plot_totals_and_means_2deg_en4():
    panels_total, panels_mean = accumulate_counts_en4(DATA_DIR, RECURSIVE, LATS2, LONS2,
                                                      depth_bands=[(0,1e9)], bins_10y=BINS)
    plot_panels(OUT_FIG_TOTAL_2DEG, panels_total,
                None,
                cbar_label="Nombre d'observations (2°×2°)",
                lat_centers=LATS2, lon_centers=LONS2, scale_type="total")
    plot_panels(OUT_FIG_MEAN_2DEG, panels_mean,
                None,
                cbar_label="Nombre moyen d'observations par an (2°×2°)",
                lat_centers=LATS2, lon_centers=LONS2, scale_type="mean")

def build_and_plot_depth_total_and_means_2deg_en4():
    for (lo_d,hi_d) in DEPTH_BANDS:
        panels_total, panels_mean = accumulate_counts_en4(DATA_DIR, RECURSIVE, LATS2, LONS2,
                                                          depth_bands=[(lo_d,hi_d)], bins_10y=BINS)
        out_total = OUT_DIR / f"en4_coverage_observations_2deg_depth_{depth_tag(lo_d,hi_d)}.png"
        out_mean  = OUT_DIR / f"en4_coverage_observations_mean_per_year_2deg_depth_{depth_tag(lo_d,hi_d)}.png"
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
    print(f"[EN4] DATA_DIR={DATA_DIR}")
    print(f"[EN4] OUT_DIR ={OUT_DIR}")
    print(f"[EN4] RECURSIVE={RECURSIVE}")
    build_and_plot_totals_and_means_2deg_en4()
    build_and_plot_depth_total_and_means_2deg_en4()
