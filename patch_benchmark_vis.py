"""
patch_benchmark_vis.py
======================
Interactive Diagnostic Viewer for Land-Cover Stratified Patch Benchmark Results.
Visualizes results produced by generate_stratified_patch_benchmark.py:
  - Left panel: 9x9 spatial patch grid overlaid with dates of change detection,
    target injection markers (P1-P4), and interactive pixel selection.
  - Right top panel: Spectral Complexity (Z-score) Satellite Image Time Series (SITS)
    with Harmonized SC-CCD harmonic fits, confidence bounds, anomalies, and break dates.
  - Right bottom panel: Landsat NIR Band (0.865 um) Surface Reflectance SITS with
    USGS LCMAP PyCCD harmonic fits, confidence bounds, and break events.
  - Controls:
      * Mouse click on 9x9 spatial grid to select any pixel.
      * Keyboard interaction:
          [M] / [m] or [Right] / [Left] : Cycle Target Material
          [F] / [f] or [Up] / [Down] / [+] / [-] : Cycle Pixel Fill Fraction
          [P] / [p] or [1]-[4] : Cycle Land-Cover Stratum (Patch Location)
          [T] / [t] : Cycle through the 4 injected target pixels (P1-P4)
          [S] / [s] : Save current view to PNG
          [H] / [h] : Print help and shortcut guide
          [Q] / [Esc] : Exit viewer
      * Interactive clickable buttons along bottom for easy navigation.
"""

import os
import sys
import math
import argparse
import datetime
from datetime import timezone
from pathlib import Path
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from scipy.stats import chi2

# Ensure single-threaded BLAS/OpenMP execution for workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Ensure script directory and parent directory are on sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import core modules
import SC_CCM_main as hccd
import pyccd_landsat_main as pccd
import ccd

# Apply compatibility patch for PyCCD coordinate descent and mode array
pccd.patch_pyccd()

# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================
DEFAULT_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
PATCH_SIZE = 13
BORDER_MARGIN = 2
INJ_OFFSETS = [
    (4, 4),  # P1
    (4, 8),  # P2
    (8, 4),  # P3
    (8, 8),  # P4
]
INJECTION_ONSET_DATE = "2022-06-01"

LAND_COVER_PATCHES = [
    {
        "id": "CROPLAND",
        "class": "Cropland",
        "name": "Rush Open Farmland",
        "r0": 1266,
        "c0": 225,
        "color": "#e6ab02",
    },
    {
        "id": "TREE COVER",
        "class": "Forest",
        "name": "Mendon Ponds Park Forest",
        "r0": 1144,
        "c0": 528,
        "color": "#2ca02c",
    },
    {
        "id": "WETLAND",
        "class": "Wetland",
        "name": "Thousand Acre Swamp (Penfield)",
        "r0": 608,
        "c0": 858,
        "color": "#17becf",
    },
    {
        "id": "DEVELOPED",
        "class": "Developed Land",
        "name": "Rochester Airport",
        "r0": 803,
        "c0": 269,
        "color": "#d62728",
    },
]


# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================
def load_benchmark_container(h5_path=DEFAULT_H5_PATH):
    """
    Loads benchmark dataset hierarchy, preloads metadata and baseline chips,
    and returns a structured dictionary along with an open HDF5 file handle.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Master HDF5 container not found at: {h5_path}")

    f = h5py.File(h5_path, "r")
    bm_group = "/STRATIFIED_CHIP_BENCHMARK"
    if bm_group not in f:
        f.close()
        raise KeyError(f"Benchmark group '{bm_group}' not found in container '{h5_path}'. "
                       f"Please run generate_stratified_patch_benchmark.py first.")

    bm = f[bm_group]
    meta_grp = bm["METADATA"]

    # Extract metadata attributes
    harm_acq_times = meta_grp.attrs["harm_acq_times"][:]
    raw_fractions = meta_grp.attrs["mixture_fractions"][:]
    if 0.0 not in raw_fractions:
        mixture_fractions = np.concatenate([[0.0], raw_fractions]).astype(np.float32)
    else:
        mixture_fractions = raw_fractions.astype(np.float32)

    injected_offsets = meta_grp.attrs.get("injected_offsets", np.array(INJ_OFFSETS, dtype=np.int32))
    onset_date_str = meta_grp.attrs.get("injection_onset_date", INJECTION_ONSET_DATE)
    if isinstance(onset_date_str, bytes):
        onset_date_str = onset_date_str.decode("utf-8")

    onset_dt = datetime.datetime.strptime(onset_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    onset_ts = onset_dt.timestamp()

    # Preload target spectra
    spec_l8 = meta_grp["target_spectra_l8"][:]
    spec_s2 = meta_grp["target_spectra_s2"][:] if "target_spectra_s2" in meta_grp else None

    # Load baseline chips
    base_grp = bm["BASELINE_CHIPS"]
    chips = {
        "HARMONIZED_z_score": base_grp["HARMONIZED_z_score"][:],
        "HARMONIZED_common_mask": base_grp["HARMONIZED_common_mask"][:],
        "HLSL30": base_grp["HLSL30"][:],
        "HLSL30_Fmask": base_grp["HLSL30_Fmask"][:],
        "HLSL30_common_mask": base_grp["HLSL30_common_mask"][:],
        "HLSS30": base_grp["HLSS30"][:] if "HLSS30" in base_grp else None,
    }

    # Extract scale factors and fill values
    l30_scale = float(base_grp["HLSL30"].attrs.get("scale_factor", 10000.0))
    l30_fill = int(base_grp["HLSL30"].attrs.get("_FillValue", -9999))

    # Resolve acquisition timestamps for Landsat-8/9 observations
    l30_times = None
    if "/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance" in f:
        l30_times = f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"].attrs.get("acquisition_time")
    if l30_times is None and "HLSL30" in base_grp and "acquisition_time" in base_grp["HLSL30"].attrs:
        l30_times = base_grp["HLSL30"].attrs["acquisition_time"]
    if l30_times is None:
        # Fallback to harmonic timestamps for Landsat acquisitions
        l30_times = harm_acq_times[:chips["HLSL30"].shape[1]]
    else:
        l30_times = l30_times[:]

    # Resolve spacecraft source tags for harmonized observations
    spacecrafts = None
    if "/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score" in f:
        harm_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"]
        if "source_spacecraft" in harm_ds.attrs:
            sc_raw = harm_ds.attrs["source_spacecraft"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sc_raw]
        elif "source_grid" in harm_ds.attrs:
            sg_raw = harm_ds.attrs["source_grid"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sg_raw]
    if spacecrafts is None and "/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask" in f:
        mask_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask"]
        if "source_spacecraft" in mask_ds.attrs:
            sc_raw = mask_ds.attrs["source_spacecraft"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sc_raw]
    if spacecrafts is None:
        spacecrafts = ["Harmonized"] * len(harm_acq_times)

    # Load baseline change detection results directly from SCCCD_Results and PYCCD_Results
    base_scccd = {}
    base_pyccd = {}
    if "SCCCD_Results" in f and "PYCCD_Results" in f:
        sc_res = f["SCCCD_Results"]
        py_res = f["PYCCD_Results"]
        base_scccd = {
            "pred": np.stack([sc_res["predicted_series"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "rmse": np.stack([sc_res["rmse_series"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "anom": np.stack([sc_res["anomaly_flags"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "date": np.stack([sc_res["change_date_timestamp"][p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "count": np.stack([sc_res["change_count"][p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
        }
        # Mask outer 2-pixel margin because SC-CCD is not evaluated on outer margin
        base_scccd["date"][:, :2, :] = np.nan
        base_scccd["date"][:, 11:, :] = np.nan
        base_scccd["date"][:, :, :2] = np.nan
        base_scccd["date"][:, :, 11:] = np.nan
        base_scccd["count"][:, :2, :] = 0
        base_scccd["count"][:, 11:, :] = 0
        base_scccd["count"][:, :, :2] = 0
        base_scccd["count"][:, :, 11:] = 0

        base_pyccd = {
            "pred": np.stack([py_res["predicted_series"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "rmse": np.stack([py_res["rmse_series"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "anom": np.stack([py_res["anomaly_flags"][:, p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "date": np.stack([py_res["change_date_timestamp"][p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
            "count": np.stack([py_res["change_count"][p["r0"]:p["r0"]+PATCH_SIZE, p["c0"]:p["c0"]+PATCH_SIZE] for p in LAND_COVER_PATCHES]),
        }
        base_pyccd["date"][:, :2, :] = np.nan
        base_pyccd["date"][:, 11:, :] = np.nan
        base_pyccd["date"][:, :, :2] = np.nan
        base_pyccd["date"][:, :, 11:] = np.nan
        base_pyccd["count"][:, :2, :] = 0
        base_pyccd["count"][:, 11:, :] = 0
        base_pyccd["count"][:, :, :2] = 0
        base_pyccd["count"][:, :, 11:] = 0

    # Load results tables
    res_grp = bm["RESULTS"]
    scccd_table = res_grp["scccd_break_table"][:]
    pyccd_table = res_grp["pyccd_break_table"][:]

    # Parse distinct target materials from break table
    materials = []
    seen_mat_ids = set()
    num_fractions = len(raw_fractions)
    num_materials = len(spec_l8)

    for m_idx in range(num_materials):
        c_idx = m_idx * num_fractions
        rows = scccd_table[scccd_table["condition_idx"] == c_idx]
        if len(rows) > 0:
            m_id = rows[0]["material_id"].decode("utf-8") if isinstance(rows[0]["material_id"], bytes) else str(rows[0]["material_id"])
            m_name = rows[0]["material_name"].decode("utf-8") if isinstance(rows[0]["material_name"], bytes) else str(rows[0]["material_name"])
        else:
            m_id = f"mat_{m_idx}"
            m_name = f"Material {m_idx + 1}"
        materials.append({
            "idx": m_idx,
            "id": m_id,
            "name": m_name,
            "spec_l8": spec_l8[m_idx],
            "spec_s2": spec_s2[m_idx] if spec_s2 is not None else None,
        })

    # Prepare harmonic design matrices
    frac_years_harm = hccd.extract_fractional_years(harm_acq_times)
    X_full_scccd = hccd.build_harmonic_matrix(frac_years_harm)

    l30_dates = [datetime.datetime.fromtimestamp(t, tz=timezone.utc).date() for t in l30_times]
    ordinal_dates_l30 = np.array([d.toordinal() for d in l30_dates], dtype=np.int64)
    X_full_pyccd = pccd.build_harmonic_basis(ordinal_dates_l30)
    nominal_thermals = np.full(len(l30_times), pccd.NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    # Reference to simulated cubes dataset (lazily sliced on demand)
    sim_z_cubes_ds = bm["SIMULATED_CUBES/sliding_volume_z_score"]

    data_bundle = {
        "h5_file": f,
        "sim_z_cubes_ds": sim_z_cubes_ds,
        "chips": chips,
        "base_scccd": base_scccd,
        "base_pyccd": base_pyccd,
        "harm_acq_times": harm_acq_times,
        "l30_times": l30_times,
        "l30_scale": l30_scale,
        "l30_fill": l30_fill,
        "spacecrafts": spacecrafts,
        "raw_fractions": raw_fractions,
        "mixture_fractions": mixture_fractions,
        "materials": materials,
        "injected_offsets": [tuple(off) for off in injected_offsets],
        "onset_date_str": onset_date_str,
        "onset_dt": onset_dt,
        "onset_ts": onset_ts,
        "scccd_table": scccd_table,
        "pyccd_table": pyccd_table,
        "frac_years_harm": frac_years_harm,
        "X_full_scccd": X_full_scccd,
        "ordinal_dates_l30": ordinal_dates_l30,
        "X_full_pyccd": X_full_pyccd,
        "nominal_thermals": nominal_thermals,
    }
    return data_bundle


def generate_patch_rgb(l30_patch, cm_patch):
    """
    Generates a natural true-color RGB composite (Red=B4, Green=B3, Blue=B2)
    using the clearest available Landsat observation for background rendering.
    """
    valid_counts = np.sum((cm_patch == 0), axis=(1, 2))
    best_t = int(np.argmax(valid_counts))

    r = l30_patch[best_t, 3].astype(np.float32) / 10000.0
    g = l30_patch[best_t, 2].astype(np.float32) / 10000.0
    b = l30_patch[best_t, 1].astype(np.float32) / 10000.0

    rgb = np.stack([r, g, b], axis=-1)
    # Contrast stretch normalized for typical surface reflectance ranges [0.0, 0.25]
    rgb = np.clip(rgb / 0.25, 0.0, 1.0)
    return rgb, best_t


# ==============================================================================
# ALGORITHMIC TIME SERIES & BREAK EVALUATION
# ==============================================================================
class BenchmarkEngine:
    """
    Encapsulates SC-CCD and USGS PyCCD evaluation with LRU caching
    to guarantee sub-millisecond interactive responsiveness during navigation.
    """
    def __init__(self, data):
        self.data = data
        self.scccd_cache = {}
        self.pyccd_cache = {}

    def get_scccd_results(self, mat_idx, frac_idx, p_idx):
        """
        Executes Harmonized SC-CCD regression across the full 9x9 patch.
        When mixture fraction == 0.0, returns baseline SC-CCD detection produced by harmonized_CCD_main.py
        with baseline spectral complexity values produced by HLST_SC_calculations.py.
        """
        fraction = float(self.data["mixture_fractions"][frac_idx])
        if fraction == 0.0:
            return {
                "z_data": self.data["chips"]["HARMONIZED_z_score"][p_idx],
                "valid_mask": (self.data["chips"]["HARMONIZED_common_mask"][p_idx] == 0),
                "predicted": self.data["base_scccd"]["pred"][p_idx],
                "rmse": self.data["base_scccd"]["rmse"][p_idx],
                "anomaly": self.data["base_scccd"]["anom"][p_idx],
                "change_date": self.data["base_scccd"]["date"][p_idx],
                "change_count": self.data["base_scccd"]["count"][p_idx],
            }

        raw_frac_idx = frac_idx - 1
        cond_idx = mat_idx * len(self.data["raw_fractions"]) + raw_frac_idx
        cache_key = (cond_idx, p_idx)
        if cache_key in self.scccd_cache:
            return self.scccd_cache[cache_key]

        z_patch = self.data["sim_z_cubes_ds"][cond_idx, p_idx]
        mask_patch = self.data["chips"]["HARMONIZED_common_mask"][p_idx]
        valid_patch = (mask_patch == 0)

        # Only evaluate the inner 9x9 region (rows 2:11, cols 2:11)
        # The outer 2-pixel margin is NOT evaluated by SC-CCD
        y_inner = z_patch[:, 2:11, 2:11]
        valid_inner = valid_patch[:, 2:11, 2:11]

        chunk_args = (
            0, 9, 9,
            y_inner, valid_inner,
            self.data["frac_years_harm"],
            self.data["X_full_scccd"],
            self.data["harm_acq_times"],
            hccd.MIN_SAMPLES,
            hccd.CHANGE_PROBABILITY,
            hccd.CONSECUTIVE_ANOMALIES
        )
        res = hccd._process_row_chunk(chunk_args)

        pred_full = np.full(z_patch.shape, np.nan, dtype=np.float32)
        rmse_full = np.full(z_patch.shape, np.nan, dtype=np.float32)
        anom_full = np.zeros(z_patch.shape, dtype=np.uint8)
        date_full = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
        count_full = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.int32)

        pred_full[:, 2:11, 2:11] = res[2]
        rmse_full[:, 2:11, 2:11] = res[3]
        anom_full[:, 2:11, 2:11] = res[5]
        date_full[2:11, 2:11] = res[6]
        count_full[2:11, 2:11] = res[7]

        result = {
            "z_data": z_patch,
            "valid_mask": valid_patch,
            "predicted": pred_full,
            "rmse": rmse_full,
            "anomaly": anom_full,
            "change_date": date_full,
            "change_count": count_full,
        }
        self.scccd_cache[cache_key] = result
        return result

    def get_pyccd_results(self, mat_idx, frac_idx, p_idx, r, c):
        """
        Executes USGS PyCCD 7-band detection on the selected pixel.
        When mixture fraction == 0.0, returns baseline PyCCD detection produced by pyccd_landsat_main.py
        with Landsat surface reflectance from /HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance.
        """
        fraction = float(self.data["mixture_fractions"][frac_idx])
        if fraction == 0.0:
            l30_patch = self.data["chips"]["HLSL30"][p_idx]
            fm_patch = self.data["chips"]["HLSL30_Fmask"][p_idx]
            cm_patch = self.data["chips"]["HLSL30_common_mask"][p_idx]

            raw_nirs = l30_patch[:, 4, r, c]
            actual_nir = raw_nirs.astype(np.float32) / float(self.data["l30_scale"])
            valid_refl = (raw_nirs != self.data["l30_fill"]) & (raw_nirs != -9999) & (raw_nirs != -32768) & (actual_nir > 0.0)
            valid_mask = ~cm_patch[:, r, c].astype(bool) & (fm_patch[:, r, c] != 255) & valid_refl

            pred_series = self.data["base_pyccd"]["pred"][p_idx, :, r, c]
            rmse_series = self.data["base_pyccd"]["rmse"][p_idx, :, r, c]
            anom_flags = self.data["base_pyccd"]["anom"][p_idx, :, r, c]
            break_date = self.data["base_pyccd"]["date"][p_idx, r, c]
            break_count = int(self.data["base_pyccd"]["count"][p_idx, r, c])

            breaks = []
            if not np.isnan(break_date) and break_count > 0:
                dt_obj = datetime.datetime.fromtimestamp(break_date, timezone.utc).date()
                breaks.append({"break_day": dt_obj.toordinal(), "change_probability": 1.0})

            return {
                "actual_nir": actual_nir,
                "valid_mask": valid_mask,
                "pred_series": pred_series,
                "rmse_series": rmse_series,
                "anomaly": anom_flags,
                "models": [],
                "breaks": breaks,
                "break_date": break_date,
                "break_count": break_count,
                "is_injected": False,
            }

        raw_frac_idx = frac_idx - 1
        cond_idx = mat_idx * len(self.data["raw_fractions"]) + raw_frac_idx
        cache_key = (cond_idx, p_idx, r, c)
        if cache_key in self.pyccd_cache:
            return self.pyccd_cache[cache_key]

        l30_patch = self.data["chips"]["HLSL30"][p_idx]
        fm_patch = self.data["chips"]["HLSL30_Fmask"][p_idx]
        cm_patch = self.data["chips"]["HLSL30_common_mask"][p_idx]

        blues = l30_patch[:, 1, r, c].copy()
        greens = l30_patch[:, 2, r, c].copy()
        reds = l30_patch[:, 3, r, c].copy()
        nirs = l30_patch[:, 4, r, c].copy()
        swir1s = l30_patch[:, 5, r, c].copy()
        swir2s = l30_patch[:, 6, r, c].copy()

        is_injected = (r, c) in self.data["injected_offsets"]
        f_weight = fraction if is_injected else 0.0

        if is_injected and f_weight > 0:
            spec_l8 = self.data["materials"][mat_idx]["spec_l8"]
            l30_times = self.data["l30_times"]
            onset_ts = self.data["onset_ts"]
            post_onset_idx = np.where(l30_times >= onset_ts)[0]
            scale = self.data["l30_scale"]
            fill_val = self.data["l30_fill"]

            for t in post_onset_idx:
                obs = np.array([blues[t], greens[t], reds[t], nirs[t], swir1s[t], swir2s[t]])
                if not np.any(obs == fill_val):
                    blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8[1] * scale))
                    greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8[2] * scale))
                    reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8[3] * scale))
                    nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8[4] * scale))
                    swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8[5] * scale))
                    swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8[6] * scale))

        fm_pixel = fm_patch[:, r, c]
        cm_pixel = cm_patch[:, r, c]
        sr_pixel_bands = np.column_stack([blues, greens, reds, nirs, swir1s, swir2s])
        qas = pccd.decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=self.data["l30_fill"])

        proc_params = {
            "QA_BITPACKED": False,
            "QA_FILL": 0,
            "QA_CLEAR": 1,
            "QA_WATER": 2,
            "QA_SHADOW": 3,
            "QA_SNOW": 4,
            "QA_CLOUD": 5,
            "LASSO_MAX_ITER": 2000,
        }

        ordinal_dates = self.data["ordinal_dates_l30"]
        num_frames = len(ordinal_dates)
        nominal_thermals = self.data["nominal_thermals"]

        try:
            ccd_res = ccd.detect(
                ordinal_dates, blues, greens, reds, nirs, swir1s, swir2s,
                nominal_thermals, qas, params=proc_params
            )
            models = ccd_res.get("change_models", [])
        except Exception:
            models = []

        breaks = [m for m in models if m.get("change_probability", 0) > 0]

        # Reconstruct NIR prediction and RMSE time series across model segments
        pred_series = np.full(num_frames, np.nan, dtype=np.float32)
        rmse_series = np.full(num_frames, np.nan, dtype=np.float32)

        if models:
            X_full = self.data["X_full_pyccd"]
            scale = self.data["l30_scale"]
            num_models = len(models)

            if num_models == 1:
                active_indices = np.zeros(num_frames, dtype=int)
            else:
                split_days = [
                    m["break_day"] if m.get("change_probability", 0) > 0 else m["end_day"]
                    for m in models[:-1]
                ]
                active_indices = np.zeros(num_frames, dtype=int)
                for t_idx, ord_d in enumerate(ordinal_dates):
                    m_idx = 0
                    while m_idx < len(split_days) and ord_d > split_days[m_idx]:
                        m_idx += 1
                    active_indices[t_idx] = m_idx

            target_preds = []
            target_rmses = []
            for m in models:
                b_info = m["nir"]
                theta = np.array([float(b_info["intercept"])] + list(b_info["coefficients"]), dtype=np.float64)
                target_preds.append((X_full @ theta) / scale)
                target_rmses.append(float(b_info["rmse"]) / scale)

            in_range = (ordinal_dates >= models[0]["start_day"]) & (ordinal_dates <= models[-1]["end_day"])
            for t_idx in range(num_frames):
                if in_range[t_idx]:
                    m_act = active_indices[t_idx]
                    pred_series[t_idx] = target_preds[m_act][t_idx]
                    rmse_series[t_idx] = target_rmses[m_act]

        actual_nir = nirs.astype(np.float32) / float(self.data["l30_scale"])
        valid_refl = (nirs != self.data["l30_fill"]) & (nirs != -9999) & (nirs != -32768) & (actual_nir > 0.0)
        valid_mask = ~cm_pixel.astype(bool) & (fm_pixel != 255) & valid_refl

        result = {
            "actual_nir": actual_nir,
            "valid_mask": valid_mask,
            "pred_series": pred_series,
            "rmse_series": rmse_series,
            "models": models,
            "breaks": breaks,
            "is_injected": is_injected,
        }
        self.pyccd_cache[cache_key] = result
        return result


# ==============================================================================
# INTERACTIVE BENCHMARK VIEWER CLASS
# ==============================================================================
class PatchBenchmarkViewer:
    """
    Main interactive visualization controller linking:
      1. Left: 9x9 Spatial change detection date grid with pixel selection.
      2. Top Right: Harmonized SC-CCD Z-score time series with harmonic fits.
      3. Bottom Right: USGS LCMAP Landsat NIR time series with PyCCD fits.
      4. Keyboard & mouse event bindings for rapid exploration.
    """
    def __init__(self, data, initial_patch=0, initial_mat=0, initial_frac=0, initial_pixel=(4, 4)):
        self.data = data
        self.engine = BenchmarkEngine(data)

        self.num_patches = len(LAND_COVER_PATCHES)
        self.num_materials = len(data["materials"])
        self.num_fractions = len(data["mixture_fractions"])

        self.p_idx = max(0, min(self.num_patches - 1, initial_patch))
        self.mat_idx = max(0, min(self.num_materials - 1, initial_mat))
        self.frac_idx = max(0, min(self.num_fractions - 1, initial_frac))
        self.sel_r, self.sel_c = initial_pixel

        # Precompute RGB composites for all 4 strata
        self.patch_rgbs = []
        for p in range(self.num_patches):
            rgb, _ = generate_patch_rgb(
                self.data["chips"]["HLSL30"][p],
                self.data["chips"]["HLSL30_common_mask"][p]
            )
            self.patch_rgbs.append(rgb)

        # Setup figure and layout
        plt.style.use("default")
        self.fig = plt.figure(figsize=(19, 10))
        self.fig.canvas.manager.set_window_title(
            "Stratified Patch Benchmark Viewer: Harmonized SC-CCD vs. USGS LCMAP PyCCD"
        )

        # GridSpec: Left (Col 0) for 9x9 spatial grid & colorbar; Right (Col 1) for stacked SITS plots
        self.gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[1.0, 1.85],
            height_ratios=[1.0, 1.0],
            left=0.05, right=0.96, top=0.91, bottom=0.10,
            hspace=0.28, wspace=0.18
        )

        gs_left = self.gs[:, 0].subgridspec(2, 1, height_ratios=[1.0, 0.045], hspace=0.10)
        self.ax_grid = self.fig.add_subplot(gs_left[0])
        self.ax_cbar = self.fig.add_subplot(gs_left[1])
        self.ax_sc = self.fig.add_subplot(self.gs[0, 1])
        self.ax_nir = self.fig.add_subplot(self.gs[1, 1], sharex=self.ax_sc)

        # Interactive state elements
        self.cbar = None
        self.rect_patch = None
        self.target_markers = []
        self.buttons = []

        self._setup_buttons()
        self._connect_events()

    def _setup_buttons(self):
        """Creates interactive navigation buttons along the bottom."""
        btn_specs = [
            ("◀ Mat", 0.05, 0.025, 0.055, 0.035, self.prev_material),
            ("Mat ▶", 0.11, 0.025, 0.055, 0.035, self.next_material),
            ("◀ Frac", 0.18, 0.025, 0.055, 0.035, self.prev_fraction),
            ("Frac ▶", 0.24, 0.025, 0.055, 0.035, self.next_fraction),
            ("◀ Patch", 0.31, 0.025, 0.058, 0.035, self.prev_patch),
            ("Patch ▶", 0.373, 0.025, 0.058, 0.035, self.next_patch),
            ("Target P1", 0.45, 0.025, 0.065, 0.035, lambda e: self.select_pixel(2, 2)),
            ("Target P2", 0.52, 0.025, 0.065, 0.035, lambda e: self.select_pixel(2, 6)),
            ("Target P3", 0.59, 0.025, 0.065, 0.035, lambda e: self.select_pixel(6, 2)),
            ("Target P4", 0.66, 0.025, 0.065, 0.035, lambda e: self.select_pixel(6, 6)),
            ("Save PNG", 0.74, 0.025, 0.070, 0.035, lambda e: self.save_png()),
        ]

        for label, left, bottom, width, height, callback in btn_specs:
            ax_btn = self.fig.add_axes([left, bottom, width, height])
            btn = Button(ax_btn, label, color="#f0f0f0", hovercolor="#d0e0ff")
            btn.on_clicked(callback)
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")
            self.buttons.append((btn, ax_btn))

    def _connect_events(self):
        """Connects mouse click and keypress handlers to matplotlib canvas."""
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    # ==========================================================================
    # NAVIGATION ACTIONS
    # ==========================================================================
    def next_material(self, event=None):
        self.mat_idx = (self.mat_idx + 1) % self.num_materials
        self.render()

    def prev_material(self, event=None):
        self.mat_idx = (self.mat_idx - 1) % self.num_materials
        self.render()

    def next_fraction(self, event=None):
        self.frac_idx = min(self.num_fractions - 1, self.frac_idx + 1)
        self.render()

    def prev_fraction(self, event=None):
        self.frac_idx = max(0, self.frac_idx - 1)
        self.render()

    def next_patch(self, event=None):
        self.p_idx = (self.p_idx + 1) % self.num_patches
        self.render()

    def prev_patch(self, event=None):
        self.p_idx = (self.p_idx - 1) % self.num_patches
        self.render()

    def cycle_target_pixel(self, event=None):
        targets = self.data["injected_offsets"]
        curr = (self.sel_r, self.sel_c)
        if curr in targets:
            idx = (targets.index(curr) + 1) % len(targets)
        else:
            idx = 0
        self.select_pixel(targets[idx][0], targets[idx][1])

    def select_pixel(self, r, c):
        if 0 <= r < PATCH_SIZE and 0 <= c < PATCH_SIZE:
            self.sel_r = r
            self.sel_c = c
            self.render(refresh_grid=False)

    def on_click(self, event):
        if event.inaxes != self.ax_grid or event.xdata is None or event.ydata is None:
            return
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if 0 <= r < PATCH_SIZE and 0 <= c < PATCH_SIZE:
            self.select_pixel(r, c)

    def on_key_press(self, event):
        key = event.key
        if key in ("m", "right"):
            self.next_material()
        elif key in ("M", "left"):
            self.prev_material()
        elif key in ("f", "up", "+", "="):
            self.next_fraction()
        elif key in ("F", "down", "-", "_"):
            self.prev_fraction()
        elif key in ("p", "]"):
            self.next_patch()
        elif key in ("P", "["):
            self.prev_patch()
        elif key in ("1", "2", "3", "4"):
            self.p_idx = int(key) - 1
            self.render()
        elif key in ("t", "T"):
            self.cycle_target_pixel()
        elif key in ("s", "S"):
            self.save_png()
        elif key in ("h", "H"):
            self.print_help()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def print_help(self):
        print("\n" + "=" * 70)
        print("STRATIFIED PATCH BENCHMARK VIEWER KEYBOARD SHORTCUTS:")
        print("  [M] / [m] or [Right] / [Left] : Cycle Target Material")
        print("  [F] / [f] or [Up] / [Down] / [+] / [-] : Cycle Pixel Fill Fraction")
        print("  [P] / [p] or [1]-[4] : Cycle Land-Cover Stratum (Patch)")
        print("  [T] / [t] : Cycle through the 4 injected target pixels (P1-P4)")
        print("  [Mouse Click] on 9x9 grid : Select pixel to inspect SITS")
        print("  [S] / [s] : Save current figure to PNG")
        print("  [Q] / [Esc] : Exit viewer")
        print("=" * 70 + "\n")

    def save_png(self, output_path=None):
        if output_path is None:
            mat_id = self.data["materials"][self.mat_idx]["id"]
            p_id = LAND_COVER_PATCHES[self.p_idx]["id"].replace(" ", "_")
            frac = self.data["mixture_fractions"][self.frac_idx]
            output_path = f"benchmark_vis_{p_id}_{mat_id}_f{frac:.2f}_r{self.sel_r}c{self.sel_c}.png"

        self.fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Screenshot successfully saved to: {os.path.abspath(output_path)}")

    # ==========================================================================
    # RENDERING ENGINE
    # ==========================================================================
    def render(self, refresh_grid=True):
        """
        Renders the entire 3-panel display:
          1. ax_grid: 9x9 patch with true-color background and break date overlay.
          2. ax_sc: Harmonized SC-CCD Z-score time series.
          3. ax_nir: Landsat NIR PyCCD surface reflectance time series.
        """
        fraction = float(self.data["mixture_fractions"][self.frac_idx])
        is_baseline = (fraction == 0.0)
        cond_idx = None if is_baseline else (self.mat_idx * len(self.data["raw_fractions"]) + (self.frac_idx - 1))
        mat_info = self.data["materials"][self.mat_idx]
        patch_info = LAND_COVER_PATCHES[self.p_idx]

        # Compute / fetch SC-CCD results across full 9x9 patch
        scccd_res = self.engine.get_scccd_results(self.mat_idx, self.frac_idx, self.p_idx)
        # Compute / fetch PyCCD results for currently selected pixel
        pyccd_res = self.engine.get_pyccd_results(
            self.mat_idx, self.frac_idx, self.p_idx, self.sel_r, self.sel_c
        )

        # ----------------------------------------------------------------------
        # 1. RENDER 9x9 SPATIAL GRID (ax_grid)
        # ----------------------------------------------------------------------
        if refresh_grid:
            self.ax_grid.clear()
            self.ax_cbar.clear()
            self.cbar = None

            # Underlay clear-sky natural true-color RGB composite
            rgb_base = self.patch_rgbs[self.p_idx]
            grid_extent = [-0.5, PATCH_SIZE - 0.5, PATCH_SIZE - 0.5, -0.5]
            self.ax_grid.imshow(rgb_base, interpolation="nearest", extent=grid_extent)

            # Overlay detected change dates using gist_rainbow colormap
            change_dates = scccd_res["change_date"].copy()
            change_counts = scccd_res["change_count"]
            change_dates[change_counts == 0] = np.nan

            if not np.all(np.isnan(change_dates)):
                self.ax_cbar.set_visible(True)
                from matplotlib.cm import gist_rainbow
                cmap = gist_rainbow
                cmap.set_bad(color="white", alpha=0.0)
                masked_dates = np.ma.masked_invalid(change_dates)
                im = self.ax_grid.imshow(
                    masked_dates, cmap=cmap, alpha=0.72,
                    interpolation="nearest", extent=grid_extent
                )
                self.cbar = self.fig.colorbar(im, cax=self.ax_cbar, orientation="horizontal")
                self.cbar.set_label("SC-CCD Detected Change Date", fontsize=9, fontweight="bold")
                ticks = self.cbar.get_ticks()
                min_t, max_t = np.nanmin(change_dates), np.nanmax(change_dates)
                if not np.isnan(min_t) and not np.isnan(max_t):
                    valid_ticks = [t for t in ticks if min_t <= t <= max_t]
                    if len(valid_ticks) > 0:
                        self.cbar.set_ticks(valid_ticks)
                        self.cbar.set_ticklabels([
                            datetime.datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
                            for t in valid_ticks
                        ], fontsize=8)
            else:
                self.ax_cbar.set_visible(False)

            # Mark inner 9x9 evaluated boundary with dashed yellow box
            eval_box = mpatches.Rectangle(
                (1.5, 1.5), 9, 9,
                linewidth=1.8, edgecolor="yellow", facecolor="none", linestyle="--",
                label="Evaluated Region (9x9)"
            )
            self.ax_grid.add_patch(eval_box)

            # Mark 4 designated target injection pixels (P1-P4)
            for p_num, (roff, coff) in enumerate(self.data["injected_offsets"], 1):
                self.ax_grid.plot(
                    coff, roff, marker="o", markersize=9,
                    markeredgecolor="white", markeredgewidth=1.8, markerfacecolor="red"
                )
                self.ax_grid.text(
                    coff + 0.22, roff - 0.22, f"P{p_num}",
                    color="white", fontweight="bold", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#1a1a1a", alpha=0.75, edgecolor="white", linewidth=0.8)
                )

            # Grid coordinates & labels (labels centered on pixels, grid aligned to pixel edges/sides)
            self.ax_grid.set_xticks(np.arange(PATCH_SIZE))
            self.ax_grid.set_yticks(np.arange(PATCH_SIZE))
            self.ax_grid.set_xticks(np.arange(-0.5, PATCH_SIZE, 1.0), minor=True)
            self.ax_grid.set_yticks(np.arange(-0.5, PATCH_SIZE, 1.0), minor=True)
            self.ax_grid.grid(False, which="major")
            self.ax_grid.grid(True, which="minor", color="white", linestyle=":", linewidth=0.8, alpha=0.6)
            self.ax_grid.tick_params(which="minor", length=0)
            self.ax_grid.set_xlim(-0.5, PATCH_SIZE - 0.5)
            self.ax_grid.set_ylim(PATCH_SIZE - 0.5, -0.5)
            self.ax_grid.set_xlabel(f"Patch Column (0 to {PATCH_SIZE - 1})", fontsize=10, fontweight="bold")
            self.ax_grid.set_ylabel(f"Patch Row (0 to {PATCH_SIZE - 1})", fontsize=10, fontweight="bold")

            if is_baseline:
                grid_title = (
                    f"{PATCH_SIZE}x{PATCH_SIZE} Spatial Grid: {patch_info['class']} ({patch_info['name']})\n"
                    f"Baseline Mode (Unperturbed) | Evaluated: Inner 9x9 (Yellow Box)"
                )
            else:
                grid_title = (
                    f"{PATCH_SIZE}x{PATCH_SIZE} Spatial Grid: {patch_info['class']} ({patch_info['name']})\n"
                    f"Target: {mat_info['name']} ({fraction:.2f}) | Evaluated: Inner 9x9 (Yellow Box)"
                )
            self.ax_grid.set_title(grid_title, fontsize=11, fontweight="bold", pad=8)

            # Selection bounding box
            self.rect_patch = mpatches.Rectangle(
                (self.sel_c - 0.5, self.sel_r - 0.5), 1, 1,
                linewidth=2.8, edgecolor="cyan", facecolor="none", linestyle="-"
            )
            self.ax_grid.add_patch(self.rect_patch)
        else:
            # Quick position update of selection box without full grid redraw
            if self.rect_patch is not None:
                self.rect_patch.set_xy((self.sel_c - 0.5, self.sel_r - 0.5))

        # ----------------------------------------------------------------------
        # 2. RENDER HARMONIZED SC-CCD TIME SERIES (ax_sc)
        # ----------------------------------------------------------------------
        self.ax_sc.clear()

        harm_times = self.data["harm_acq_times"]
        harm_dates = np.array([datetime.datetime.fromtimestamp(t, timezone.utc) for t in harm_times])
        actual_z = scccd_res["z_data"][:, self.sel_r, self.sel_c]
        valid_mask_z = scccd_res["valid_mask"][:, self.sel_r, self.sel_c]
        pred_z = scccd_res["predicted"][:, self.sel_r, self.sel_c]
        rmse_z = scccd_res["rmse"][:, self.sel_r, self.sel_c]
        anom_z = (scccd_res["anomaly"][:, self.sel_r, self.sel_c] == 1)

        spacecrafts = np.array(self.data["spacecrafts"])
        markers = [
            ("s", "Sentinel", ["sentinel", "s30", "hlss30"]),
            ("o", "Landsat", ["landsat", "l30", "hlsl30"]),
            ("D", "Tanager", ["tanager"])
        ]

        # Plot observations differentiated by sensor platform
        matched_any = np.zeros(len(spacecrafts), dtype=bool)
        for marker_type, sc_keyword, sc_aliases in markers:
            sc_mask = np.array([any(alias in str(s).lower() for alias in sc_aliases) for s in spacecrafts])
            matched_any |= sc_mask

            idx_valid_norm = valid_mask_z & sc_mask & ~anom_z
            if np.any(idx_valid_norm):
                self.ax_sc.plot(
                    harm_dates[idx_valid_norm], actual_z[idx_valid_norm],
                    color="black", marker=marker_type, markersize=3.5, linestyle="None",
                    label=f"Valid ({sc_keyword})"
                )

            idx_valid_anom = valid_mask_z & sc_mask & anom_z
            if np.any(idx_valid_anom):
                self.ax_sc.plot(
                    harm_dates[idx_valid_anom], actual_z[idx_valid_anom],
                    color="crimson", marker=marker_type, markersize=5.5, linestyle="None",
                    label=f"Anomaly ({sc_keyword})"
                )

            idx_invalid = ~valid_mask_z & sc_mask
            if np.any(idx_invalid):
                self.ax_sc.plot(
                    harm_dates[idx_invalid], actual_z[idx_invalid],
                    color="gray", marker=marker_type, markersize=2.5, linestyle="None",
                    markerfacecolor="none", alpha=0.35, label=f"Masked ({sc_keyword})"
                )

        # Plot any remaining unclassified platforms if present
        if not np.all(matched_any):
            sc_mask = ~matched_any
            idx_valid_norm = valid_mask_z & sc_mask & ~anom_z
            if np.any(idx_valid_norm):
                self.ax_sc.plot(
                    harm_dates[idx_valid_norm], actual_z[idx_valid_norm],
                    color="black", marker="x", markersize=3.5, linestyle="None",
                    label="Valid (Other)"
                )
            idx_valid_anom = valid_mask_z & sc_mask & anom_z
            if np.any(idx_valid_anom):
                self.ax_sc.plot(
                    harm_dates[idx_valid_anom], actual_z[idx_valid_anom],
                    color="crimson", marker="x", markersize=5.5, linestyle="None",
                    label="Anomaly (Other)"
                )
            idx_invalid = ~valid_mask_z & sc_mask
            if np.any(idx_invalid):
                self.ax_sc.plot(
                    harm_dates[idx_invalid], actual_z[idx_invalid],
                    color="gray", marker="x", markersize=2.5, linestyle="None",
                    markerfacecolor="none", alpha=0.35, label="Masked (Other)"
                )

        # Plot SC-CCD Harmonic Regression Segments with confidence bands
        p_val = hccd.CHANGE_PROBABILITY
        df = 3
        rmse_mult = float(np.sqrt(chi2.ppf(p_val, df=df)))
        bound_label = f"Prediction \u00b1{rmse_mult:.2f}\u03c3 (p={p_val})"

        pred_mask = ~np.isnan(pred_z)
        if np.any(pred_mask):
            pred_dates = harm_dates[pred_mask]
            preds = pred_z[pred_mask]
            rmses = rmse_z[pred_mask]

            rmse_diff = np.diff(rmses)
            break_indices = np.where(rmse_diff != 0)[0] + 1

            date_segs = np.split(pred_dates, break_indices)
            pred_segs = np.split(preds, break_indices)
            rmse_segs = np.split(rmses, break_indices)

            seg_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
            for i, (s_dates, s_preds, s_rmses) in enumerate(zip(date_segs, pred_segs, rmse_segs)):
                if len(s_dates) == 0:
                    continue
                c = seg_colors[i % len(seg_colors)]
                upper = s_preds + rmse_mult * s_rmses
                lower = s_preds - rmse_mult * s_rmses

                seg_lbl = "Harmonic Fit" if i == 0 else f"Harmonic Fit (Seg {i+1})"
                self.ax_sc.plot(s_dates, s_preds, color=c, linestyle="--", linewidth=1.8, label=seg_lbl)
                self.ax_sc.fill_between(
                    s_dates, lower, upper, color=c, alpha=0.15,
                    label=bound_label if i == 0 else None
                )

        # Mark injection onset date
        self.ax_sc.axvline(
            x=self.data["onset_dt"], color="darkviolet", linestyle="--", linewidth=1.8,
            label=f"Onset ({self.data['onset_date_str']})"
        )

        # Mark detected SC-CCD break date
        sc_break_ts = scccd_res["change_date"][self.sel_r, self.sel_c]
        sc_break_cnt = int(scccd_res["change_count"][self.sel_r, self.sel_c])
        if not np.isnan(sc_break_ts) and sc_break_cnt > 0:
            dt_break = datetime.datetime.fromtimestamp(sc_break_ts, timezone.utc)
            if not is_baseline:
                latency_sc = (sc_break_ts - self.data["onset_ts"]) / 86400.0
                sc_lbl = f"SC-CCD Break: {dt_break.strftime('%Y-%m-%d')} (+{latency_sc:.1f}d)"
            else:
                sc_lbl = f"Baseline Break: {dt_break.strftime('%Y-%m-%d')}"
            self.ax_sc.axvline(
                x=dt_break, color="crimson", linestyle=":", linewidth=2.2,
                label=sc_lbl
            )

        # Pixel identification string
        is_inj_target = (self.sel_r, self.sel_c) in self.data["injected_offsets"]
        target_tag = f"Target P{self.data['injected_offsets'].index((self.sel_r, self.sel_c)) + 1}" if is_inj_target else "Background / Neighbor"

        if is_baseline:
            sc_title = (
                f"Harmonized SC-CCD: Baseline Spectral Complexity (Z-Score) | Pixel ({self.sel_r}, {self.sel_c}) [{target_tag}] | "
                f"Breaks: {sc_break_cnt}"
            )
        else:
            sc_title = (
                f"Harmonized SC-CCD: Spectral Complexity (Z-Score) | Pixel ({self.sel_r}, {self.sel_c}) [{target_tag}] | "
                f"Breaks: {sc_break_cnt}"
            )
        self.ax_sc.set_title(sc_title, fontsize=11, fontweight="bold", pad=6)
        self.ax_sc.set_ylabel("Spectral Complexity (Z-Score)", fontsize=10, fontweight="bold")
        self.ax_sc.grid(True, linestyle=":", alpha=0.6)

        # Deduplicate and place legend
        handles, labels = self.ax_sc.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_sc.legend(
            by_label.values(), by_label.keys(),
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8
        )

        # ----------------------------------------------------------------------
        # 3. RENDER LANDSAT NIR PYCCD TIME SERIES (ax_nir)
        # ----------------------------------------------------------------------
        self.ax_nir.clear()

        l30_times = self.data["l30_times"]
        l30_dates = np.array([datetime.datetime.fromtimestamp(t, timezone.utc) for t in l30_times])
        actual_nir = pyccd_res["actual_nir"]
        valid_mask_nir = pyccd_res["valid_mask"]
        pred_nir = pyccd_res["pred_series"]
        rmse_nir = pyccd_res["rmse_series"]
        models_py = pyccd_res["models"]
        breaks_py = pyccd_res["breaks"]

        # Plot Landsat NIR surface reflectance observations
        # Filter out invalid pixel values (e.g. fill values at -1 / -0.9999 or non-positive nodata)
        is_valid_refl_val = (actual_nir > 0.0) & ~np.isclose(actual_nir, -1.0, atol=0.05) & (actual_nir != self.data["l30_fill"] / float(self.data["l30_scale"]))

        idx_norm_nir = valid_mask_nir & is_valid_refl_val
        if np.any(idx_norm_nir):
            self.ax_nir.plot(
                l30_dates[idx_norm_nir], actual_nir[idx_norm_nir],
                color="black", marker="o", markersize=3.8, linestyle="None",
                label="Valid (Landsat NIR)"
            )

        idx_mask_nir = ~valid_mask_nir & is_valid_refl_val
        if np.any(idx_mask_nir):
            self.ax_nir.plot(
                l30_dates[idx_mask_nir], actual_nir[idx_mask_nir],
                color="silver", marker="o", markersize=2.5, linestyle="None",
                markerfacecolor="none", alpha=0.4, label="Masked (Cloud/Shadow)"
            )

        # Plot CCDC Harmonic Segments with +/- 3 sigma confidence envelope
        pred_mask_nir = ~np.isnan(pred_nir)
        if np.any(pred_mask_nir):
            pred_dates = l30_dates[pred_mask_nir]
            preds = pred_nir[pred_mask_nir]
            rmses = rmse_nir[pred_mask_nir]

            rmse_diff = np.diff(rmses)
            break_indices = np.where(rmse_diff != 0)[0] + 1

            date_segs = np.split(pred_dates, break_indices)
            pred_segs = np.split(preds, break_indices)
            rmse_segs = np.split(rmses, break_indices)

            seg_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
            for i, (s_dates, s_preds, s_rmses) in enumerate(zip(date_segs, pred_segs, rmse_segs)):
                if len(s_dates) == 0:
                    continue
                c = seg_colors[i % len(seg_colors)]
                upper = s_preds + 3.0 * s_rmses
                lower = s_preds - 3.0 * s_rmses

                lbl = "PyCCD Fit" if i == 0 else f"PyCCD Fit (Seg {i+1})"
                self.ax_nir.plot(s_dates, s_preds, color=c, linestyle="-", linewidth=2.0, label=lbl)
                self.ax_nir.fill_between(
                    s_dates, lower, upper, color=c, alpha=0.15,
                    label="PyCCD \u00b13\u03c3" if i == 0 else None
                )

        # Mark injection onset date
        self.ax_nir.axvline(
            x=self.data["onset_dt"], color="darkviolet", linestyle="--", linewidth=1.8,
            label=f"Onset ({self.data['onset_date_str']})"
        )

        # Mark detected PyCCD breaks
        first_break_str = "None"
        py_break_ts = pyccd_res.get("break_date")
        py_break_cnt = int(pyccd_res.get("break_count", len(breaks_py)))

        if is_baseline:
            if py_break_ts is not None and not np.isnan(py_break_ts) and py_break_cnt > 0:
                dt_py_break = datetime.datetime.fromtimestamp(py_break_ts, timezone.utc)
                self.ax_nir.axvline(
                    x=dt_py_break, color="crimson", linestyle=":", linewidth=2.2,
                    label=f"Baseline Break: {dt_py_break.strftime('%Y-%m-%d')}"
                )
        else:
            if breaks_py:
                for b_idx, b in enumerate(breaks_py):
                    b_day = b["break_day"]
                    if b_day > 0:
                        dt_py_break = datetime.date.fromordinal(b_day)
                        dt_py_ts = datetime.datetime(dt_py_break.year, dt_py_break.month, dt_py_break.day, tzinfo=timezone.utc).timestamp()
                        latency_py = (dt_py_ts - self.data["onset_ts"]) / 86400.0
                        lbl = f"PyCCD Break: {dt_py_break.strftime('%Y-%m-%d')} (+{latency_py:.1f}d)" if b_idx == 0 else None
                        if b_idx == 0:
                            first_break_str = f"{dt_py_break.strftime('%Y-%m-%d')} (+{latency_py:.1f}d)"
                        self.ax_nir.axvline(
                            x=datetime.datetime(dt_py_break.year, dt_py_break.month, dt_py_break.day, tzinfo=timezone.utc),
                            color="crimson", linestyle=":", linewidth=2.2, label=lbl
                        )

        if is_baseline:
            nir_title = (
                f"USGS LCMAP Landsat PyCCD: Baseline NIR (0.865 \u03bcm) | Pixel ({self.sel_r}, {self.sel_c}) [{target_tag}] | "
                f"Breaks: {py_break_cnt}"
            )
        else:
            nir_title = (
                f"USGS LCMAP Landsat PyCCD (NIR Band 0.865 \u03bcm) | Pixel ({self.sel_r}, {self.sel_c}) [{target_tag}] | "
                f"Breaks: {py_break_cnt}"
            )
        self.ax_nir.set_title(nir_title, fontsize=11, fontweight="bold", pad=6)
        self.ax_nir.set_ylabel("NIR Surface Reflectance", fontsize=10, fontweight="bold")
        self.ax_nir.set_xlabel("Acquisition Date", fontsize=10, fontweight="bold")
        self.ax_nir.grid(True, linestyle=":", alpha=0.6)

        # Shared X-axis date formatting
        self.ax_nir.xaxis.set_major_locator(mdates.YearLocator())
        self.ax_nir.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        self.ax_nir.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

        # Deduplicate and place legend
        handles, labels = self.ax_nir.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_nir.legend(
            by_label.values(), by_label.keys(),
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8
        )

        # ----------------------------------------------------------------------
        # 4. SUPER-TITLE & BENCHMARK QUANTITATIVE METRIC BANNER
        # ----------------------------------------------------------------------
        # Look up tabular benchmark metrics if selected pixel is one of the 4 targets
        bench_sc_info = "N/A (Neighbor)"
        bench_py_info = "N/A (Neighbor)"
        if is_baseline:
            bench_sc_info = f"Baseline Clean ({sc_break_cnt} breaks)"
            bench_py_info = f"Baseline Clean ({py_break_cnt} breaks)"
            super_str = (
                f"STRATIFIED PATCH BENCHMARK | Stratum: [{self.p_idx + 1}/{self.num_patches}] {patch_info['class']} ({patch_info['name']}) | "
                f"Mode: Baseline (Unperturbed HLST/LCMAP) | Fill Fraction: 0.00 (Baseline)\n"
                f"Selected Pixel: Row {self.sel_r}, Col {self.sel_c} [{target_tag}]  ⟶  "
                f"SC-CCD: {bench_sc_info}  |  PyCCD: {bench_py_info}"
            )
        else:
            if is_inj_target:
                p_num = self.data["injected_offsets"].index((self.sel_r, self.sel_c)) + 1
                sc_tbl = self.data["scccd_table"]
                py_tbl = self.data["pyccd_table"]

                sc_match = sc_tbl[
                    (sc_tbl["condition_idx"] == cond_idx) &
                    (sc_tbl["patch_id"] == patch_info["id"].encode("utf-8")) &
                    (sc_tbl["pixel_num"] == p_num)
                ]
                if len(sc_match) > 0:
                    row = sc_match[0]
                    det = row["break_detected"]
                    lat = row["latency_days"]
                    mag = row["break_magnitude"]
                    bench_sc_info = f"Detected: {det}" + (f" (+{lat:.1f}d, mag={mag:.2f})" if det else "")

                py_match = py_tbl[
                    (py_tbl["condition_idx"] == cond_idx) &
                    (py_tbl["patch_id"] == patch_info["id"].encode("utf-8")) &
                    (py_tbl["pixel_num"] == p_num)
                ]
                if len(py_match) > 0:
                    row = py_match[0]
                    det = row["break_detected"]
                    lat = row["latency_days"]
                    mag = row["break_magnitude"]
                    bench_py_info = f"Detected: {det}" + (f" (+{lat:.1f}d, mag={mag:.2f})" if det else "")

            super_str = (
                f"STRATIFIED PATCH BENCHMARK | Stratum: [{self.p_idx + 1}/{self.num_patches}] {patch_info['class']} ({patch_info['name']}) | "
                f"Target: [{self.mat_idx + 1}/{self.num_materials}] {mat_info['name']} | "
                f"Fraction: [{self.frac_idx + 1}/{self.num_fractions}] {fraction:.2f}\n"
                f"Selected Pixel: Row {self.sel_r}, Col {self.sel_c} [{target_tag}]  ⟶  "
                f"SC-CCD: {bench_sc_info}  |  PyCCD: {bench_py_info}"
            )
        self.fig.suptitle(super_str, fontsize=12, fontweight="bold", y=0.975)

        self.fig.canvas.draw_idle()


# ==============================================================================
# MAIN COMMAND LINE INTERFACE & ENTRY POINT
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Interactive Diagnostic Viewer for Stratified Patch Target Injection Benchmark Results."
    )
    parser.add_argument(
        "--h5-path", type=str, default=DEFAULT_H5_PATH,
        help=f"Path to master HDF5 container (default: {DEFAULT_H5_PATH})"
    )
    parser.add_argument(
        "--patch", type=str, default="0",
        help="Initial patch index (0-3) or stratum ID (e.g. CROPLAND, TREE COVER, WETLAND, DEVELOPED)"
    )
    parser.add_argument(
        "--material", type=str, default="0",
        help="Initial material index (0-19) or ID (e.g. concrete, asphalt shingle, etc.)"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.0,
        help="Initial mixture fill fraction (default: 0.0 for unperturbed baseline)"
    )
    parser.add_argument(
        "--pixel", type=int, nargs=2, default=[4, 4],
        help="Initial pixel row and col (default: 4 4 for target P1)"
    )
    parser.add_argument(
        "--save-png", type=str, default=None,
        help="Filename to export initial rendering to PNG and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Disable interactive GUI display (useful for batch or headless environments)"
    )
    return parser.parse_args()


def resolve_indices(args, data):
    """Maps CLI string or index arguments to valid numeric indices."""
    # Resolve patch index
    p_arg = str(args.patch).strip().upper()
    patch_idx = 0
    if p_arg.isdigit():
        patch_idx = int(p_arg)
    else:
        for idx, p in enumerate(LAND_COVER_PATCHES):
            if p_arg in (p["id"].upper(), p["class"].upper(), p["name"].upper()):
                patch_idx = idx
                break

    # Resolve material index
    m_arg = str(args.material).strip().lower()
    mat_idx = 0
    if m_arg.isdigit():
        mat_idx = int(m_arg)
    else:
        for idx, m in enumerate(data["materials"]):
            if m_arg in (m["id"].lower(), m["name"].lower()):
                mat_idx = idx
                break

    # Resolve fraction index
    frac_val = float(args.fraction)
    fractions = data["mixture_fractions"]
    frac_idx = int(np.argmin(np.abs(fractions - frac_val)))

    # Resolve pixel coordinates
    sel_r, sel_c = args.pixel[0], args.pixel[1]
    return patch_idx, mat_idx, frac_idx, (sel_r, sel_c)


def main():
    args = parse_arguments()

    if args.save_png or args.no_gui:
        matplotlib.use("Agg")

    print(f"\n[Patch Benchmark Viewer] Loading benchmark container: {args.h5_path}")
    data = load_benchmark_container(args.h5_path)

    p_idx, mat_idx, frac_idx, (sel_r, sel_c) = resolve_indices(args, data)

    print(f"Loaded {len(data['materials'])} materials across {len(LAND_COVER_PATCHES)} strata.")
    print(f"Initializing viewer at:")
    print(f"  Stratum : [{p_idx}] {LAND_COVER_PATCHES[p_idx]['class']} ({LAND_COVER_PATCHES[p_idx]['name']})")
    print(f"  Material: [{mat_idx}] {data['materials'][mat_idx]['name']}")
    print(f"  Fraction: [{frac_idx}] {data['mixture_fractions'][frac_idx]:.2f}")
    print(f"  Pixel   : Row {sel_r}, Col {sel_c}")

    viewer = PatchBenchmarkViewer(
        data,
        initial_patch=p_idx,
        initial_mat=mat_idx,
        initial_frac=frac_idx,
        initial_pixel=(sel_r, sel_c)
    )
    viewer.render(refresh_grid=True)

    if args.save_png:
        viewer.save_png(args.save_png)
        print(f"Export complete. Exiting...")
        data["h5_file"].close()
        return

    if not args.no_gui:
        viewer.print_help()
        plt.show()

    data["h5_file"].close()


if __name__ == "__main__":
    main()
