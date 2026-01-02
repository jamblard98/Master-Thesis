#!/usr/bin/env python3
"""
extract_nemo_by_decade.py

Extrait NEMO aux positions des OBSERVATIONS HISTORIQUES (EN4/WOD) contenues dans les CSV
produits par yamazaki_en4_wod_comparison_v3.py.

Format de sortie (1 obs historique × 19 années récentes = 19 lignes):
  source, hist_year, hist_month, hist_day, hist_lat, hist_lon, hist_depth_m,
  hist_temperature, yamazaki_T, nemo_hist_T, recent_year, nemo_recent_T

MÉTHODOLOGIE "PLUS PROCHE VOISIN" (conforme à ton screenshot) :
- Horizontal (Eq. 1) : on choisit (i,j) qui minimise
    d(i,j) = sqrt( (lat_ij - lat_obs)^2 + (lon_ij - lon_obs)^2 )
- Vertical (Eq. 2) : une fois (i,j) trouvé, on choisit k qui minimise
    |depth_k - depth_obs|

IMPORTANT :
- Le point (lat_obs, lon_obs, depth_obs) utilisé pour NN est celui de l'observation historique
  (hist_lat, hist_lon, hist_depth_m) — pas Yamazaki.
- Longitudes : cohérence en [-180, 180]. On utilise une différence "wrap" pour éviter les erreurs
  près de +/-180.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import gc

print("="*80)
print("EXTRACTION NEMO PAR DÉCENNIE")
print("="*80)

DECADES = [
    (1900, 1909), (1910, 1919), (1920, 1929), (1930, 1939),
    (1940, 1949), (1950, 1959), (1960, 1969)
]
RECENT_YEARS = list(range(2005, 2024))  # 2005-2023

FILLVALUE_NEMO = 1.e+20  # thetao:_FillValue / missing_value


def parse_args():
    p = argparse.ArgumentParser(description="Extraction NEMO par décennie")
    p.add_argument('--yamazaki-csv-dir', type=str, required=True,
                   help="Dossier CSV Yamazaki (yamazaki_en4_wod_DJF_*.csv)")
    p.add_argument('--nemo-nc', type=str, required=True,
                   help="Fichier NetCDF NEMO")
    p.add_argument('--out-dir', type=str, required=True,
                   help="Dossier sortie")
    return p.parse_args()


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


def load_yamazaki_csv_decade(csv_dir, decade_start, decade_end):
    """Charge le CSV décennal contenant les observations historiques + Yamazaki."""
    csv_file = Path(csv_dir) / f"yamazaki_en4_wod_DJF_{decade_start}_{decade_end}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_file}")

    df = pd.read_csv(csv_file, low_memory=False, na_filter=True, keep_default_na=True)

    for c in ["hist_year", "hist_month", "hist_day", "hist_lat", "hist_lon", "hist_depth_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        df["hist_year"].notna() &
        df["hist_month"].notna() &
        df["hist_lat"].notna() &
        df["hist_lon"].notna() &
        df["hist_depth_m"].notna()
    ].copy()

    # sécuriser convention lon
    df["hist_lon"] = normalize_lon(df["hist_lon"].to_numpy())

    print(f"   ✓ {len(df):,} observations chargées")
    return df


def load_nemo(nc_path):
    """Charge NetCDF NEMO + index temporel (year, month) -> t_idx."""
    print(f"\n📂 Chargement NEMO: {nc_path}")
    ds = xr.open_dataset(nc_path, decode_times=False)

    time_sec = ds["time_counter"].values.astype(float)

    # calendrier 365_day
    years = 1900 + (time_sec / (365.0 * 24.0 * 3600.0)).astype(int)
    days = time_sec / (24.0 * 3600.0)
    months = (((days % 365.0) / 30.4167).astype(int) % 12) + 1  # 1..12

    lats = ds["nav_lat"].values.astype(float)
    lons = normalize_lon(ds["nav_lon"].values.astype(float))  # cohérence [-180,180)
    depths = ds["deptht"].values.astype(float)
    temp = ds["thetao"].values

    print(f"   ✓ NEMO: {len(years)} pas de temps, {len(depths)} profondeurs")
    print(f"   ✓ Grille: {lats.shape}")
    print(f"   ✓ Années: {years.min():.0f}-{years.max():.0f}")
    print(f"   ✓ Longitudes NEMO normalisées en [-180,180) en interne")

    time_index = {}
    for t_idx in range(len(years)):
        y = int(years[t_idx])
        m = int(months[t_idx])
        time_index.setdefault(y, {}).setdefault(m, []).append(t_idx)

    return {
        'ds': ds,
        'lats': lats,
        'lons': lons,
        'depths': depths,
        'temp': temp,
        'time_index': time_index
    }


def _valid_temp(v):
    """Valide une valeur thetao en tenant compte du FillValue."""
    try:
        fv = float(v)
    except Exception:
        return False
    return np.isfinite(fv) and (fv != FILLVALUE_NEMO)


def extract_nemo_for_decade(df_decade, nemo, out_dir, decade_start, decade_end):
    """Extrait NEMO pour une décennie et exporte CSV (chunked)."""

    print(f"\n{'='*80}")
    print(f"DÉCENNIE {decade_start}-{decade_end}")
    print(f"{'='*80}")

    out_csv = Path(out_dir) / f"nemo_yamazaki_DJF_{decade_start}_{decade_end}.csv"

    lats = nemo['lats']
    lons = nemo['lons']
    depths = nemo['depths']
    temp = nemo['temp']
    time_index = nemo['time_index']

    count_valid_hist = 0
    count_no_nemo_hist = 0

    CHUNK_SIZE = 5000
    total_obs = len(df_decade)
    n_chunks = (total_obs + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"📦 Traitement par chunks de {CHUNK_SIZE:,} observations")
    print(f"📦 Nombre de chunks: {n_chunks}")

    first_write = True

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_obs)

        df_chunk = df_decade.iloc[start_idx:end_idx]
        print(f"\n   📦 Chunk {chunk_idx+1}/{n_chunks}: lignes {start_idx:,} à {end_idx:,}")

        rows = []

        for _, row in df_chunk.iterrows():
            # Observation historique (PRIME pour NN)
            obs_year = int(row['hist_year'])
            obs_month = int(row['hist_month'])
            obs_lat = float(row['hist_lat'])
            obs_lon = float(row['hist_lon'])         # déjà normalisé [-180,180)
            obs_depth = float(row['hist_depth_m'])

            # -------------------------
            # Eq (1) : NN horizontal (lat/lon) sur la grille NEMO curvilinéaire
            # d^2 = (lat_ij - lat_obs)^2 + (lon_ij - lon_obs)^2 (avec wrap sur lon)
            # -------------------------
            dlat = (lats - obs_lat)
            dlon = lon_diff_deg(lons, obs_lon)
            dist2 = dlat**2 + dlon**2
            y_idx, x_idx = np.unravel_index(np.argmin(dist2), lats.shape)

            # -------------------------
            # Eq (2) : NN vertical
            # -------------------------
            depth_idx = int(np.argmin(np.abs(depths - obs_depth)))

            # NEMO historique (même année, même mois)
            nemo_hist_T = np.nan
            if obs_year in time_index and obs_month in time_index[obs_year]:
                t_indices = time_index[obs_year][obs_month]
                if t_indices:
                    t_idx = t_indices[0]
                    temp_val = temp[t_idx, depth_idx, y_idx, x_idx]
                    if _valid_temp(temp_val):
                        nemo_hist_T = float(temp_val)
                        count_valid_hist += 1
                    else:
                        count_no_nemo_hist += 1
                else:
                    count_no_nemo_hist += 1
            else:
                count_no_nemo_hist += 1

            # logique identique à avant : skip si pas de NEMO historique valide
            if not np.isfinite(nemo_hist_T):
                continue

            # NEMO récent (2005-2023), même mois et même point
            for recent_year in RECENT_YEARS:
                nemo_recent_T = np.nan

                if recent_year in time_index and obs_month in time_index[recent_year]:
                    t_indices_recent = time_index[recent_year][obs_month]
                    if t_indices_recent:
                        t_idx = t_indices_recent[0]
                        temp_val = temp[t_idx, depth_idx, y_idx, x_idx]
                        if _valid_temp(temp_val):
                            nemo_recent_T = float(temp_val)

                row_dict = row.to_dict()
                row_dict['nemo_hist_T'] = nemo_hist_T
                row_dict['recent_year'] = recent_year
                row_dict['nemo_recent_T'] = nemo_recent_T
                rows.append(row_dict)

        if rows:
            df_out = pd.DataFrame(rows)
            if first_write:
                df_out.to_csv(out_csv, index=False, mode='w')
                first_write = False
            else:
                df_out.to_csv(out_csv, index=False, mode='a', header=False)

            print(f"      ✓ {len(rows):,} lignes écrites")
            del df_out
            del rows

        pct = 100.0 * end_idx / total_obs
        print(f"      📊 Progression totale: {end_idx:,}/{total_obs:,} ({pct:.1f}%)")

    print(f"\n   ✓ {count_valid_hist:,} observations avec NEMO historique valide")
    if count_no_nemo_hist > 0:
        print(f"   ⚠️  {count_no_nemo_hist:,} observations sans NEMO historique (exclues)")

    if out_csv.exists():
        size_mb = out_csv.stat().st_size / (1024**2)
        print(f"\n   ✅ CSV créé: {out_csv.name}")
        print(f"      Taille: {size_mb:.1f} MB")
    else:
        print(f"\n   ⚠️  Aucun CSV écrit (aucune obs NEMO historique valide)")

    return count_valid_hist * len(RECENT_YEARS)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nemo = load_nemo(args.nemo_nc)

    total_rows = 0

    for decade_start, decade_end in DECADES:
        print(f"\n{'='*80}")
        print(f"TRAITEMENT DÉCENNIE {decade_start}-{decade_end}")
        print(f"{'='*80}")

        df_decade = None
        try:
            df_decade = load_yamazaki_csv_decade(
                args.yamazaki_csv_dir, decade_start, decade_end
            )

            n_rows = extract_nemo_for_decade(
                df_decade, nemo, out_dir, decade_start, decade_end
            )
            total_rows += n_rows

        except FileNotFoundError as e:
            print(f"   ⏭️  Skip: {e}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

        if df_decade is not None:
            del df_decade
        gc.collect()
        print(f"   🧹 Mémoire libérée")

    nemo['ds'].close()

    print(f"\n{'='*80}")
    print("✅ EXTRACTION TERMINÉE")
    print(f"{'='*80}")
    print(f"\n✅ {total_rows:,} lignes estimées au total")
    print(f"✅ 7 fichiers CSV créés dans {out_dir}")
    print(f"\n🎯 Fichiers prêts pour nemo_and_histo_profiles.py")


if __name__ == "__main__":
    main()
