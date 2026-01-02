#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yamazaki_en4_wod_comparison_v3.py  (MODIFIÉ: "Yamazaki-only" pour la référence récente)
--------------------------------------------------------------------------------------
Script robuste pour comparer EN4/WOD (1900-1969) avec Yamazaki (climatologie mensuelle).

CHANGEMENT CLÉ (conforme à ce qu'on a acté) :
- On NE collecte PLUS d'observations "récentes" EN4/WOD (2005-2025).
- La "référence récente" utilisée partout est uniquement Yamazaki :
  pour chaque observation historique conservée après déduplication,
  on échantillonne Yamazaki au plus proche voisin (lat/lon/depth) pour le même mois.

MÉTHODOLOGIE "PLUS PROCHE VOISIN" (conforme à ton screenshot) :
- Horizontal (Eq. 1) : on choisit (i,j) qui minimise
    d(i,j) = sqrt( (lat_ij - lat_obs)^2 + (lon_ij - lon_obs)^2 )
- Vertical (Eq. 2) : une fois (i,j) trouvé, on choisit k qui minimise
    |depth_k - depth_obs|

CONTRAINTES :
- Longitudes : on travaille en [-180,180]. On utilise une différence en longitude "wrap"
  (distance minimale) pour éviter des erreurs près de +/-180.
- Les colonnes des CSV exportés restent IDENTIQUES (mêmes noms) :
  - CSV principal : recent_count / recent_mean / recent_std conservées mais neutralisées.
"""

import argparse
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
from netCDF4 import num2date, Dataset


# ============================================================================
# Configuration
# ============================================================================

SEASON_MONTHS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'ALL': list(range(1, 13)),
}

DECADES = [
    (1900, 1909), (1910, 1919), (1920, 1929), (1930, 1939),
    (1940, 1949), (1950, 1959), (1960, 1969)
]

# Tolérances strictes pour déduplication
STRICT_LAT_TOL = 0.01      # ~1 km
STRICT_LON_TOL = 0.01      # ~1 km
STRICT_DEPTH_TOL = 0.5     # m
STRICT_TEMP_TOL = 0.28     # °C

# Filtres géographiques et temporels
LAT_MIN = -90.0
LAT_MAX = -60.0
DEPTH_MIN = 10.0
DEPTH_MAX = 200.0
HIST_Y0 = 1900
HIST_Y1 = 1969


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def normalize_lon(lon):
    """Normalise longitude à [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def lon_diff_deg(lon_a, lon_b):
    """
    Différence minimale (wrap) en degrés entre deux longitudes.
    Retour dans [-180, 180).
    """
    return normalize_lon(np.asarray(lon_a, dtype=float) - float(lon_b))

def parse_cf_time(time_vals, units, calendar):
    """Parse CF time convention."""
    try:
        dates = num2date(time_vals, units=units, calendar=calendar)
        dates = np.atleast_1d(dates)
        years = np.array([d.year for d in dates], dtype=int)
        months = np.array([d.month for d in dates], dtype=int)
        days = np.array([d.day for d in dates], dtype=int)
        return years, months, days
    except Exception as e:
        print(f"⚠️  Erreur parse temps : {e}")
        shape = np.shape(time_vals) or (1,)
        fill = np.full(shape, -999, dtype=int)
        return fill, fill, fill


# ============================================================================
# Yamazaki Climatologie
# ============================================================================

class YamazakiClim:
    """Classe pour échantillonner la climatologie Yamazaki."""

    def __init__(self, nc_path):
        print(f"📂 Chargement Yamazaki: {nc_path}")
        self.ds = xr.open_dataset(str(nc_path), decode_times=False)

        self.lats = np.asarray(self.ds["latitude"].values, dtype=float)     # (nlat,)
        self.lons = np.asarray(self.ds["longitude"].values, dtype=float)    # (nlon,) attendu [-180,180]
        self.deps = np.asarray(self.ds["depth"].values, dtype=float)        # (ndep,)
        self.temp = np.asarray(self.ds["temp"].values, dtype=float)         # (dep, month, lat, lon)
        self.err  = np.asarray(self.ds["temp_err"].values, dtype=float)     # (dep, month, lat, lon)

        # Sécuriser convention lon interne (au cas où)
        self.lons = normalize_lon(self.lons)

        print(f"   ✓ Yamazaki chargé: {len(self.lats)} lats, {len(self.lons)} lons, {len(self.deps)} depths")

    def sample(self, month, lat, lon, depth):
        """
        Échantillonne Yamazaki au point le plus proche selon :
          - Eq (1) horizontal : minimise sqrt((dlat)^2 + (dlon)^2)
          - Eq (2) vertical   : minimise |depth_k - depth_obs|
        Retourne (temp, err, dist_lat, dist_lon, dist_depth)
        où dist_* = distance aux coordonnées du point de grille le plus proche.
        """
        m = int(month) - 1
        if m < 0 or m > 11:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        lon = float(normalize_lon(lon))

        # Vérifier limites lat/depth (lon : on accepte via wrap, mais on garde une barrière simple)
        if depth < float(np.nanmin(self.deps)) or depth > float(np.nanmax(self.deps)):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        if lat < float(np.nanmin(self.lats)) or lat > float(np.nanmax(self.lats)):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

        # -------------------------
        # Eq (1) : nearest neighbor horizontal (lat/lon)
        # d^2 = (lat_i - lat_obs)^2 + (lon_j - lon_obs)^2
        # avec dlon "wrap"
        # -------------------------
        dlat = (self.lats - float(lat))                    # (nlat,)
        dlon = lon_diff_deg(self.lons, lon)                # (nlon,)
        dist2 = dlat[:, None]**2 + dlon[None, :]**2        # (nlat, nlon)

        i_lat, i_lon = np.unravel_index(np.argmin(dist2), dist2.shape)
        i_lat = int(i_lat)
        i_lon = int(i_lon)

        # -------------------------
        # Eq (2) : nearest neighbor vertical
        # -------------------------
        i_dep = int(np.argmin(np.abs(self.deps - float(depth))))

        dist_lat = float(abs(self.lats[i_lat] - float(lat)))
        dist_lon = float(abs(lon_diff_deg(self.lons[i_lon], lon)))
        dist_depth = float(abs(self.deps[i_dep] - float(depth)))

        T = float(self.temp[i_dep, m, i_lat, i_lon])
        E = float(self.err[i_dep, m, i_lat, i_lon])

        return (T, E, dist_lat, dist_lon, dist_depth)

    def close(self):
        self.ds.close()


# ============================================================================
# Lecture EN4
# ============================================================================

def read_en4_profiles(nc_path, year_min, year_max, season_months,
                      lat_min, lat_max, depth_min, depth_max):
    """
    Lit un fichier EN4 et retourne les profils filtrés.
    Retourne: liste de dicts {year, month, day, lat, lon, depth, temp, source}
    """
    profiles = []

    try:
        ds = Dataset(str(nc_path), 'r')

        lat_var = None
        lon_var = None
        time_var = None
        depth_var = None
        temp_var = None
        qc_var = None

        for v in ['LATITUDE', 'latitude', 'LAT', 'lat']:
            if v in ds.variables:
                lat_var = v
                break

        for v in ['LONGITUDE', 'longitude', 'LON', 'lon']:
            if v in ds.variables:
                lon_var = v
                break

        for v in ['JULD', 'TIME', 'time', 'juld']:
            if v in ds.variables:
                time_var = v
                break

        for v in ['DEPH_CORRECTED', 'DEPH', 'DEPTH', 'depth']:
            if v in ds.variables:
                depth_var = v
                break

        for v in ['POTM_CORRECTED', 'TEMP', 'temperature', 'TEMPERATURE']:
            if v in ds.variables:
                temp_var = v
                break

        for v in ['POTM_CORRECTED_QC', 'TEMP_QC', 'TEMPERATURE_QC']:
            if v in ds.variables:
                qc_var = v
                break

        if not all([lat_var, lon_var, time_var, depth_var, temp_var]):
            ds.close()
            return profiles

        lats = np.asarray(ds.variables[lat_var][:], dtype=float)
        lons = np.asarray(ds.variables[lon_var][:], dtype=float)
        times = np.asarray(ds.variables[time_var][:])
        depths = np.asarray(ds.variables[depth_var][:], dtype=float)
        temps = np.asarray(ds.variables[temp_var][:], dtype=float)

        time_units = ds.variables[time_var].units if hasattr(ds.variables[time_var], 'units') else 'days since 1950-01-01'
        time_cal = ds.variables[time_var].calendar if hasattr(ds.variables[time_var], 'calendar') else 'gregorian'
        years, months, days = parse_cf_time(times, time_units, time_cal)

        qc = None
        if qc_var:
            qc = np.asarray(ds.variables[qc_var][:])

        if temps.ndim == 1:
            for i in range(len(temps)):
                if years[i] < year_min or years[i] > year_max:
                    continue
                if months[i] not in season_months:
                    continue
                if lats[i] < lat_min or lats[i] > lat_max:
                    continue
                if depths[i] < depth_min or depths[i] > depth_max:
                    continue
                if not np.isfinite(temps[i]):
                    continue

                if qc is not None:
                    qc_val = qc[i]
                    if qc_val not in [b'0', b'1', 0, 1, '0', '1']:
                        continue

                profiles.append({
                    'year': int(years[i]),
                    'month': int(months[i]),
                    'day': int(days[i]),
                    'lat': float(lats[i]),
                    'lon': float(normalize_lon(lons[i])),
                    'depth': float(depths[i]),
                    'temp': float(temps[i]),
                    'source': 'EN4'
                })

        elif temps.ndim == 2:
            n_prof = temps.shape[0]

            if lats.ndim == 0:
                lats = np.full(n_prof, float(lats))
            if lons.ndim == 0:
                lons = np.full(n_prof, float(lons))

            for i_prof in range(n_prof):
                if years[i_prof] < year_min or years[i_prof] > year_max:
                    continue
                if months[i_prof] not in season_months:
                    continue
                if lats[i_prof] < lat_min or lats[i_prof] > lat_max:
                    continue

                for i_lev in range(temps.shape[1]):
                    d = depths[i_prof, i_lev] if depths.ndim == 2 else depths[i_lev]
                    t = temps[i_prof, i_lev]

                    if not np.isfinite(d) or not np.isfinite(t):
                        continue
                    if d < depth_min or d > depth_max:
                        continue

                    if qc is not None:
                        qc_val = qc[i_prof, i_lev] if qc.ndim == 2 else qc[i_prof]
                        if qc_val not in [b'0', b'1', 0, 1, '0', '1']:
                            continue

                    profiles.append({
                        'year': int(years[i_prof]),
                        'month': int(months[i_prof]),
                        'day': int(days[i_prof]),
                        'lat': float(lats[i_prof]),
                        'lon': float(normalize_lon(lons[i_prof])),
                        'depth': float(d),
                        'temp': float(t),
                        'source': 'EN4'
                    })

        ds.close()

    except Exception as e:
        print(f"⚠️  Erreur lecture {nc_path.name}: {e}")

    return profiles


# ============================================================================
# Lecture WOD
# ============================================================================

def read_wod_profiles(nc_path, year_min, year_max, season_months,
                      lat_min, lat_max, depth_min, depth_max):
    """
    Lit un fichier WOD et retourne les profils filtrés.
    Retourne: liste de dicts {year, month, day, lat, lon, depth, temp, source}
    """
    profiles = []

    try:
        ds = Dataset(str(nc_path), 'r')

        if 'lat' not in ds.variables or 'lon' not in ds.variables:
            ds.close()
            return profiles

        lats = np.asarray(ds.variables['lat'][:], dtype=float)
        lons = np.asarray(ds.variables['lon'][:], dtype=float)
        times = np.asarray(ds.variables['time'][:])

        time_units = ds.variables['time'].units if hasattr(ds.variables['time'], 'units') else 'days since 1770-01-01'
        time_cal = ds.variables['time'].calendar if hasattr(ds.variables['time'], 'calendar') else 'standard'
        years, months, days = parse_cf_time(times, time_units, time_cal)

        temps = np.asarray(ds.variables['Temperature'][:], dtype=float)
        temp_flags = np.asarray(ds.variables['Temperature_WODflag'][:])
        temp_rows = np.asarray(ds.variables['Temperature_row_size'][:], dtype=int)

        depths = np.asarray(ds.variables['z'][:], dtype=float)
        depth_rows = np.asarray(ds.variables['z_row_size'][:], dtype=int)

        temp_offsets = np.r_[0, np.cumsum(temp_rows)[:-1]]
        depth_offsets = np.r_[0, np.cumsum(depth_rows)[:-1]]

        lons = normalize_lon(lons)

        for i_prof in range(len(lats)):
            if years[i_prof] < year_min or years[i_prof] > year_max:
                continue
            if months[i_prof] not in season_months:
                continue
            if lats[i_prof] < lat_min or lats[i_prof] > lat_max:
                continue

            t0 = int(temp_offsets[i_prof])
            tN = t0 + int(temp_rows[i_prof])
            z0 = int(depth_offsets[i_prof])
            zN = z0 + int(depth_rows[i_prof])

            n_pairs = min(tN - t0, zN - z0)
            if n_pairs <= 0:
                continue

            prof_temps = temps[t0:t0 + n_pairs]
            prof_flags = temp_flags[t0:t0 + n_pairs]
            prof_depths = depths[z0:z0 + n_pairs]

            for j in range(n_pairs):
                if prof_flags[j] != 0:
                    continue
                if not np.isfinite(prof_depths[j]) or not np.isfinite(prof_temps[j]):
                    continue
                if prof_depths[j] < depth_min or prof_depths[j] > depth_max:
                    continue

                profiles.append({
                    'year': int(years[i_prof]),
                    'month': int(months[i_prof]),
                    'day': int(days[i_prof]),
                    'lat': float(lats[i_prof]),
                    'lon': float(lons[i_prof]),
                    'depth': float(prof_depths[j]),
                    'temp': float(prof_temps[j]),
                    'source': 'WOD'
                })

        ds.close()

    except Exception as e:
        print(f"⚠️  Erreur lecture {nc_path.name}: {e}")

    return profiles


# ============================================================================
# Déduplication stricte
# ============================================================================

def deduplicate_profiles(profiles):
    """
    Déduplique les profils selon critères stricts:
    - Même jour + lat/lon (±0.01°) + profondeur (±0.5m)
    - Si températures dans une enveloppe <= STRICT_TEMP_TOL → garder 1 (priorité EN4)
    - Sinon → supprimer tous
    """
    print(f"🔧 Déduplication de {len(profiles):,} profils...")

    grouped = defaultdict(list)

    for prof in profiles:
        key = (
            prof['year'],
            prof['month'],
            prof['day'],
            round(prof['lat'] / STRICT_LAT_TOL) * STRICT_LAT_TOL,
            round(prof['lon'] / STRICT_LON_TOL) * STRICT_LON_TOL,
            round(prof['depth'] / STRICT_DEPTH_TOL) * STRICT_DEPTH_TOL
        )
        grouped[key].append(prof)

    deduplicated = []
    n_removed_diff_temps = 0
    n_removed_duplicates = 0

    for _, group in grouped.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue

        temps = [p['temp'] for p in group]
        temp_min = min(temps)
        temp_max = max(temps)

        if (temp_max - temp_min) <= STRICT_TEMP_TOL:
            en4_profiles = [p for p in group if p['source'] == 'EN4']
            if en4_profiles:
                deduplicated.append(en4_profiles[0])
            else:
                deduplicated.append(group[0])
            n_removed_duplicates += len(group) - 1
        else:
            n_removed_diff_temps += len(group)

    print(f"   ✓ {len(deduplicated):,} profils conservés")
    print(f"   ✗ {n_removed_duplicates:,} doublons retirés (T compatibles)")
    print(f"   ✗ {n_removed_diff_temps:,} retirés (T différentes)")

    return deduplicated


# ============================================================================
# Collecte observations historiques par décennie
# ============================================================================

def collect_historical_decade(en4_dir, wod_dir, decade_start, decade_end,
                              season_months, lat_min, lat_max,
                              depth_min, depth_max):
    """
    Collecte toutes les observations historiques pour une décennie.
    Retourne: liste de profils dédupliqués
    """
    print(f"\n{'='*80}")
    print(f"📅 DÉCENNIE {decade_start}-{decade_end}")
    print(f"{'='*80}")

    all_profiles = []

    # EN4
    print(f"\n📂 Lecture EN4 ({decade_start}-{decade_end})...")
    en4_path = Path(en4_dir)
    en4_count = 0

    for year in range(decade_start, decade_end + 1):
        year_dir = en4_path / f"EN.4.2.2.profiles.c14.{year}"
        if not year_dir.exists():
            continue

        for nc_file in sorted(year_dir.glob("EN.4.2.2.f.profiles.*.nc")):
            profiles = read_en4_profiles(
                nc_file, decade_start, decade_end, season_months,
                lat_min, lat_max, depth_min, depth_max
            )
            all_profiles.extend(profiles)
            en4_count += len(profiles)

            if len(all_profiles) % 50000 == 0 and len(all_profiles) > 0:
                print(f"   ... {len(all_profiles):,} profils collectés")

    print(f"   ✓ EN4: {en4_count:,} profils")

    # WOD
    print(f"\n📂 Lecture WOD ({decade_start}-{decade_end})...")
    wod_path = Path(wod_dir)
    wod_count = 0

    for nc_file in sorted(wod_path.glob("ocldb*.nc")):
        if 'nc3' in str(nc_file):
            continue

        profiles = read_wod_profiles(
            nc_file, decade_start, decade_end, season_months,
            lat_min, lat_max, depth_min, depth_max
        )
        all_profiles.extend(profiles)
        wod_count += len(profiles)

        if len(all_profiles) % 50000 == 0:
            print(f"   ... {len(all_profiles):,} profils collectés")

    print(f"   ✓ WOD: {wod_count:,} profils")
    print(f"\n📊 Total collecté: {len(all_profiles):,} profils")

    deduplicated = deduplicate_profiles(all_profiles)

    del all_profiles
    gc.collect()

    return deduplicated


# ============================================================================
# Enrichissement Yamazaki
# ============================================================================

def enrich_with_yamazaki(profiles, yamazaki):
    """
    Ajoute les valeurs Yamazaki à chaque profil (nearest neighbor Eq.1 + Eq.2).
    """
    print(f"\n🌡️  Enrichissement Yamazaki...")

    dist_lats = []
    dist_lons = []
    dist_depths = []

    for prof in profiles:
        yam_temp, yam_err, d_lat, d_lon, d_depth = yamazaki.sample(
            prof['month'], prof['lat'], prof['lon'], prof['depth']
        )
        prof['yamazaki_T'] = yam_temp
        prof['yamazaki_err'] = yam_err
        prof['yamazaki_dist_lat'] = d_lat
        prof['yamazaki_dist_lon'] = d_lon
        prof['yamazaki_dist_depth'] = d_depth

        if np.isfinite(d_lat):
            dist_lats.append(d_lat)
        if np.isfinite(d_lon):
            dist_lons.append(d_lon)
        if np.isfinite(d_depth):
            dist_depths.append(d_depth)

    n_with_yam = sum(1 for p in profiles if np.isfinite(p.get('yamazaki_T', np.nan)))
    print(f"   ✓ {n_with_yam:,} / {len(profiles):,} profils avec Yamazaki")

    if dist_lats:
        print(f"\n   📊 Distances Yamazaki (plus proche voisin):")
        print(f"      Latitude  : max={max(dist_lats):.4f}°, mean={np.mean(dist_lats):.4f}°, median={np.median(dist_lats):.4f}°")
        print(f"      Longitude : max={max(dist_lons):.4f}°, mean={np.mean(dist_lons):.4f}°, median={np.median(dist_lons):.4f}°")
        print(f"      Profondeur: max={max(dist_depths):.2f}m, mean={np.mean(dist_depths):.2f}m, median={np.median(dist_depths):.2f}m")

    return profiles


# ============================================================================
# Export CSV (mêmes colonnes qu'avant)
# ============================================================================

def export_decade_csv(profiles, out_dir, decade_start, decade_end, season):
    """
    Exporte CSV principal: 1 ligne par observation historique (avec Yamazaki)
    + colonnes recent_* conservées mais neutralisées (0/NaN).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"yamazaki_en4_wod_{season}_{decade_start}_{decade_end}"
    csv_main = out_dir / f"{prefix}.csv"

    print(f"\n💾 Export CSV...")
    print(f"   Principal: {csv_main.name}")

    profiles_valid = [p for p in profiles if np.isfinite(p.get('yamazaki_T', np.nan))]

    n_drop = len(profiles) - len(profiles_valid)
    if n_drop > 0:
        print(f"   ⚠️  {n_drop:,} profils exclus (pas de valeur Yamazaki valide / hors grille)")

    with open(csv_main, 'w') as f:
        f.write("source,hist_year,hist_month,hist_day,hist_lat,hist_lon,hist_depth_m,")
        f.write("hist_temperature,yamazaki_T,yamazaki_err,")
        f.write("yamazaki_dist_lat,yamazaki_dist_lon,yamazaki_dist_depth,")
        f.write("recent_count,recent_mean,recent_std\n")

        for i, prof in enumerate(profiles_valid):
            f.write(f"{prof['source']},")
            f.write(f"{prof['year']},{prof['month']},{prof['day']},")
            f.write(f"{prof['lat']:.6f},{prof['lon']:.6f},{prof['depth']:.2f},")
            f.write(f"{prof['temp']:.4f},")

            yam_T = prof.get('yamazaki_T', np.nan)
            yam_err = prof.get('yamazaki_err', np.nan)
            yam_d_lat = prof.get('yamazaki_dist_lat', np.nan)
            yam_d_lon = prof.get('yamazaki_dist_lon', np.nan)
            yam_d_depth = prof.get('yamazaki_dist_depth', np.nan)

            f.write(f"{yam_T:.4f}," if np.isfinite(yam_T) else ",")
            f.write(f"{yam_err:.4f}," if np.isfinite(yam_err) else ",")
            f.write(f"{yam_d_lat:.6f}," if np.isfinite(yam_d_lat) else ",")
            f.write(f"{yam_d_lon:.6f}," if np.isfinite(yam_d_lon) else ",")
            f.write(f"{yam_d_depth:.2f}," if np.isfinite(yam_d_depth) else ",")

            recent_count = 0
            recent_mean = np.nan
            recent_std = np.nan

            f.write(f"{recent_count},")
            f.write(f"{recent_mean:.4f}," if np.isfinite(recent_mean) else ",")
            f.write(f"{recent_std:.4f}\n" if np.isfinite(recent_std) else "\n")

            if (i + 1) % 10000 == 0:
                print(f"   ... {i+1:,} / {len(profiles_valid):,} lignes écrites (principal)")

    print(f"   ✓ CSV principal: {len(profiles_valid):,} lignes")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Comparaison EN4/WOD (historique) avec Yamazaki (référence) - export CSV décennaux"
    )
    ap.add_argument("--en4-dir", type=str, required=True,
                    help="Répertoire EN4 (ex: ~/Thesis/EN4/Datas)")
    ap.add_argument("--wod-dir", type=str, required=True,
                    help="Répertoire WOD (ex: ~/Thesis/WOD/Datas)")
    ap.add_argument("--yamazaki-nc", type=str, required=True,
                    help="Fichier NetCDF Yamazaki")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Répertoire de sortie")
    ap.add_argument("--season", type=str, default="DJF",
                    choices=list(SEASON_MONTHS.keys()))
    ap.add_argument("--start-decade", type=int, default=1900,
                    help="Décennie de départ (ex: 1920 pour skip 1900-1919)")

    args = ap.parse_args()

    season_months = SEASON_MONTHS[args.season]

    print("="*80)
    print("EN4/WOD HISTORIQUES vs YAMAZAKI (NEAREST NEIGHBOR Eq.1+Eq.2) — EXPORT CSV")
    print("="*80)
    print(f"Saison: {args.season}")
    print(f"Période historique: {HIST_Y0}-{HIST_Y1}")
    print(f"Profondeur: {DEPTH_MIN}-{DEPTH_MAX}m")
    print(f"Latitude: {LAT_MIN}°S - {LAT_MAX}°S")
    print("Référence (récente): Yamazaki uniquement (pas d'extraction EN4/WOD 2005-2025)")
    print("Méthode NN: Eq(1) horizontal lat/lon, puis Eq(2) profondeur")
    print("="*80)

    yamazaki = YamazakiClim(args.yamazaki_nc)

    for decade_start, decade_end in DECADES:
        if decade_start < args.start_decade:
            print(f"\n⏭️  Skip décennie {decade_start}-{decade_end} (déjà traitée)")
            continue

        historical_profiles = collect_historical_decade(
            args.en4_dir, args.wod_dir, decade_start, decade_end,
            season_months, LAT_MIN, LAT_MAX, DEPTH_MIN, DEPTH_MAX
        )

        if len(historical_profiles) == 0:
            print(f"⚠️  Aucune observation pour {decade_start}-{decade_end}, skip")
            continue

        historical_profiles = enrich_with_yamazaki(historical_profiles, yamazaki)

        export_decade_csv(
            historical_profiles,
            args.out_dir, decade_start, decade_end, args.season
        )

        del historical_profiles
        gc.collect()

    yamazaki.close()

    print("\n" + "="*80)
    print("✅ TERMINÉ")
    print("="*80)


if __name__ == "__main__":
    main()
