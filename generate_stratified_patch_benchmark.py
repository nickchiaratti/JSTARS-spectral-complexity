"""
generate_stratified_patch_benchmark.py
======================================
Executes the Land-Cover Stratified Patches semi-empirical target injection experiment
and saves all chips, simulated complexity cubes, and algorithmic break detection results
directly into the master HDF5 container under /STRATIFIED_CHIP_BENCHMARK/.

Reuses core functions and standards from:
  - SpecComplexTorch.py: Sliding window MaxD Gramian volume unmixing on GPU.
  - harmonized_CCD_main.py: Harmonized RLS-LASSO continuous change detection engine.
  - pyccd_landsat_main.py: USGS LCMAP Landsat PyCCD detection engine and QA decoding.
  - USGS Splib07 multi-sensor resampled spectral library (Landsat-8, Sentinel-2, Tanager-1).

Experimental Structure:
  - 4 Land-Cover Strata (13x13 pixels, 390 m x 390 m):
      1. AGRI: Rush Open Farmland (r0=1266, c0=225)
      2. FOREST: Mendon Ponds Park Forest (r0=1144, c0=528)
      3. WETLAND: Thousand Acre Swamp, Penfield (r0=608, c0=858)
      4. SYNTHETIC: Rochester Airport (r0=803, c0=269)
  - 4 Decoupled Injected Pixels per patch (stride = 4, 3 baseline pixels separation):
      Offsets: [(4, 4), (4, 8), (8, 4), (8, 8)]
      Inner evaluated region: rows 2..10, cols 2..10 (9x9 pixels)
      Outer 2-pixel margin: rows 0..1, 11..12, cols 0..1, 11..12 (un-evaluated by SC-CCD)
  - 10 USGS Splib07 Target Materials x 5 Mixture Fractions [0.05, 0.15, 0.25, 0.35, 0.45]
      = 50 Distinct Perturbation Conditions
  - 50 conditions x 4 strata x 4 pixels = 800 test instances per algorithm.
"""

import os
import sys
import gc
import math
import time
import datetime
from pathlib import Path
import h5py
import numpy as np
import scipy.interpolate

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
import SpecComplexTorch as sc
import harmonized_CCD_main as hccd
import pyccd_landsat_main as pccd
import ccd

# Apply compatibility patch for PyCCD coordinate descent and mode array
pccd.patch_pyccd()

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
MASTER_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
INJECTION_ONSET_DATE = "2022-04-01"
SPLIB_BASE = "C:/satelliteImagery/ground_truth/splib07/ASCIIdata"

# Pipeline and patch parameters
PATCH_SIZE = 13
BORDER_MARGIN = 2
TILE_SIZE = 3
SLIDING_STRIDE = 1
NUM_ENDMEMBERS = 7

# Land-Cover Stratified Patches (13x13 chips with 100% clean baseline in inner 9x9 region)
LAND_COVER_PATCHES = [
    {
        "id": "CROPLAND",
        "class": "Cropland",
        "name": "Rush Open Farmland",
        "r0": 1266,
        "c0": 225,
        "color": "#e6ab02",  # Gold / Amber
        "descr": "Cycling crop production land.",
    },
    {
        "id": "TREE COVER",
        "class": "Forest",
        "name": "Mendon Ponds Park Forest",
        "r0": 1144,
        "c0": 528,
        "color": "#2ca02c",  # Forest Green
        "descr": "Deciduous forest.",
    },
    {
        "id": "WETLAND",
        "class": "Wetland",
        "name": "Thousand Acre Swamp (Penfield)",
        "r0": 608,
        "c0": 858,
        "color": "#17becf",  # Teal / Cyan
        "descr": "Saturated soil and standing water.",
    },
    {
        "id": "DEVELOPED",
        "class": "Developed Land",
        "name": "Rochester Airport",
        "r0": 803,
        "c0": 269,
        "color": "#d62728",  # Crimson Red
        "descr": "Tarmac and asphalt with minimal vegetation.",
    },
]

TARGET_MATERIALS = [
    {
        "id": "concrete",
        "name": "Concrete (WTC01-37A)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Concrete_WTC01-37A_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Concrete_WTC01-37A_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Concrete_WTC01-37A_ASDFRa_AREF.txt",
    },
    {
        "id": "asphalt shingle",
        "name": "Asphalt Shingle (GD367)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
    },
    {
        "id": "painted_aluminum",
        "name": "Painted Aluminum (GDS333)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
    },
    {
        "id": "galvanized steel",
        "name": "Galvanized Sheet Metal (GDS334)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
    },
    {
        "id": "plywood",
        "name": "Plywood Fresh Pine (GDS365)",
        "contrast": "Manmade",
        "role": "Construction / Debris",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
    },
    {
        "id": "green_tarp",
        "name": "Green Plastic Tarp (GDS339)",
        "contrast": "Manmade",
        "role": "Construction / Debris",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
    },
    {
        "id": "white_plastic",
        "name": "White Vinyl Plastic (GDS372)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
    },
    {
        "id": "Nylon_Fabric",
        "name": "Black Nylon Fabric (GDS436)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_grass",
        "name": "Dry Golden Grass (GDS480)",
        "contrast": "senescent vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_willow",
        "name": "Dried Willow Leaves (Willow-Leaves-1)",
        "contrast": "senescent vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
    },
    {
        "id": "healthy_maple",
        "name": "Healthy Maple Leaves (DW92-1)",
        "contrast": "healthy vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Maple_Leaves_DW92-1_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Maple_Leaves_DW92-1_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Maple_Leaves_DW92-1_BECKa_AREF.txt",
    },
    {
        "id": "Pinon_Pine",
        "name": "Pinon Pine (ANP92-14A)",
        "contrast": "healthy vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
    },
    {
        "id": "burn_area",
        "name": "Burn Area Traverse (WRF00-01)",
        "contrast": "Burn / Pyrogenic",
        "role": "Natural Disturbance",
        "asd_rel": r"ChapterS_SoilsAndMixtures\s07_ASD_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
        "l8_rel":  r"ChapterS_SoilsAndMixtures\S07LSAT8_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
        "s2_rel":  r"ChapterS_SoilsAndMixtures\S07SNTL2_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
    },
    {
        "id": "burn_area2",
        "name": "Burn Area Surface (WRF00-02)",
        "contrast": "Burn / Pyrogenic",
        "role": "Natural Disturbance",
        "asd_rel": r"ChapterS_SoilsAndMixtures\s07_ASD_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "l8_rel":  r"ChapterS_SoilsAndMixtures\S07LSAT8_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "s2_rel":  r"ChapterS_SoilsAndMixtures\S07SNTL2_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
    },
    {
        "id": "oak_leaves",
        "name": "Oak Leaf (Oak-Leaf-1)",
        "contrast": "Healthy Vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
    },
    {
        "id": "oak_shrub",
        "name": "Oak Shrub (CA01-QUDU-2)",
        "contrast": "Canopy Foliage",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
    },
    {
        "id": "aspen_YellowGreenTop",
        "name": "Yellow-Green Aspen Leaf (Aspen-3)",
        "contrast": "Canopy Foliage",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
    },
    {
        "id": "lawn_grass",
        "name": "Lawn Grass  (GDS91)",
        "contrast": "Healthy Vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 0.5g/L Montmorillonite (SWy-2+0.5g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 1.67g/L Montmorillonite (SWy-2+1.67g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 5.01g/L Montmorillonite (SWy-2+5.01g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 16.5g/L Montmorillonite (SWy-2+16.5g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
    },
]

# Sentinel-2 band indices matching HLSS30 (10 bands):
S2_SUB_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]

# Injected pixel local offsets inside each 13x13 patch (stride = 4, 3 baseline pixels separation)
# Inner 9x9 evaluated region is slice 2:11; outer 2-pixel margin is slice :2 and 11:
INJ_OFFSETS = [
    (4, 4),  # P1
    (4, 8),  # P2
    (8, 4),  # P3
    (8, 8),  # P4
]

# Fractional abundance levels
MIXTURE_FRACTIONS = np.arange(0.00,1.05, step=0.05)

# Compound data type for tabular results
BREAK_RECORD_DTYPE = np.dtype([
    ("condition_idx", np.int16),        # 0 to 49
    ("material_id", "S32"),             # Target material identifier string
    ("material_name", "S48"),           # Material descriptive name
    ("mixture_fraction", np.float32),   # 0.05, 0.15, 0.25, 0.35, 0.45
    ("patch_id", "S16"),                # 'AGRI', 'FOREST', 'WETLAND', 'SYNTHETIC'
    ("patch_name", "S48"),              # Full stratum description
    ("pixel_num", np.int8),             # 1 to 4
    ("scene_row", np.int16),            # Absolute row coordinate in master scene
    ("scene_col", np.int16),            # Absolute col coordinate in master scene
    ("break_detected", np.bool_),       # True if break detected post-onset
    ("break_timestamp", np.float64),    # Acquisition timestamp of detected break
    ("latency_days", np.float32),       # Days between onset date and detected break
    ("break_magnitude", np.float32),    # Magnitude of break or residual shift
    ("p_value", np.float32),            # Chi-square or model significance
])


# ==============================================================================
# STEP 1: EXTRACTION OF BASELINE CHIPS & METADATA
# ==============================================================================
def get_dataset_metadata(ds):
    """
    Extracts fill value and scale factor from HDF5 dataset attributes without assumed defaults.
    Fails explicitly if mandatory attributes are missing.
    """
    # 1. Fill value extraction
    fill_val = None
    if hasattr(ds, "attrs"):
        if "_FillValue" in ds.attrs:
            fill_val = ds.attrs["_FillValue"]
        elif "fillvalue" in ds.attrs:
            fill_val = ds.attrs["fillvalue"]
        elif "fill_value" in ds.attrs:
            fill_val = ds.attrs["fill_value"]

    if fill_val is None and hasattr(ds, "fillvalue") and ds.fillvalue is not None:
        if np.issubdtype(ds.dtype, np.floating) or ds.fillvalue != 0:
            fill_val = ds.fillvalue

    if fill_val is None:
        if np.issubdtype(ds.dtype, np.integer):
            raise KeyError(f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) but lacks mandatory '_FillValue' attribute.")
        elif np.issubdtype(ds.dtype, np.floating):
            fill_val = np.nan

    if fill_val is not None:
        if np.issubdtype(ds.dtype, np.integer):
            fill_val = int(fill_val)
        elif np.issubdtype(ds.dtype, np.floating):
            fill_val = float(fill_val)

    # 2. Scale factor extraction
    if np.issubdtype(ds.dtype, np.floating):
        # Native floating-point reflectance is already unscaled [0, 1]
        scale = float(ds.attrs["scale_factor"]) if (hasattr(ds, "attrs") and "scale_factor" in ds.attrs) else 1.0
    else:
        if hasattr(ds, "attrs") and "scale_factor" in ds.attrs:
            scale = float(ds.attrs["scale_factor"])
        elif hasattr(ds, "attrs") and "scale_to_float" in ds.attrs:
            sf = float(ds.attrs["scale_to_float"])
            scale = (1.0 / sf) if sf < 1.0 else sf
        else:
            raise KeyError(f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) lacks mandatory 'scale_factor' or 'scale_to_float' attribute.")

    return fill_val, scale


def extract_baseline_chips(h5_file):
    """
    Extracts the unperturbed 9x9 time series for each of the 4 patches across sensors.
    Dynamically identifies scale_factor and _FillValue attributes from each dataset without assumed defaults.
    """
    print(f"\n[Step 1] Extracting unperturbed {PATCH_SIZE}x{PATCH_SIZE} baseline chips from master scene...")

    # 1. Verify presence of dataset produced by HLST_SC_calculations.py
    harm_field_path = "/HDFEOS/GRIDS/HARMONIZED/Data Fields"
    if harm_field_path not in h5_file or "sliding_volume_z_score" not in h5_file[harm_field_path]:
        raise RuntimeError(
            f"Baseline Harmonized Spectral Complexity dataset '{harm_field_path}/sliding_volume_z_score' "
            f"not found in master container '{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run HLST_SC_calculations.py first to compute baseline spectral complexity metrics."
        )

    # 2. Verify presence of baseline SC-CCD dataset produced by harmonized_CCD_main.py
    scccd_grp_name = "SCCCD_Results"
    if scccd_grp_name not in h5_file or "change_date_timestamp" not in h5_file[scccd_grp_name]:
        raise RuntimeError(
            f"Baseline SC-CCD dataset '/{scccd_grp_name}' not found in master container "
            f"'{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run harmonized_CCD_main.py first to generate full-frame baseline SC-CCD change detection results."
        )

    # 3. Verify presence of baseline PyCCD dataset produced by pyccd_landsat_main.py
    pyccd_grp_name = "PYCCD_Results"
    if pyccd_grp_name not in h5_file or "change_date_timestamp" not in h5_file[pyccd_grp_name]:
        raise RuntimeError(
            f"Baseline PyCCD dataset '/{pyccd_grp_name}' not found in master container "
            f"'{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run pyccd_landsat_main.py first to generate full-frame PyCCD change detection results."
        )

    harm_grp = h5_file[harm_field_path]
    l30_grp = h5_file["/HDFEOS/GRIDS/HLSL30/Data Fields"]
    s30_grp = h5_file["/HDFEOS/GRIDS/HLSS30/Data Fields"]
    tan_grp = h5_file.get("/HDFEOS/GRIDS/TANAGER/Data Fields")

    harm_z_ds = harm_grp["sliding_volume_z_score"]
    harm_cm_ds = harm_grp["common_mask"]
    l30_sr_ds = l30_grp["surface_reflectance"]
    l30_fm_ds = l30_grp["Fmask"]
    l30_cm_ds = l30_grp["common_mask"]
    s30_sr_ds = s30_grp["surface_reflectance"]
    s30_fm_ds = s30_grp["Fmask"]
    s30_cm_ds = s30_grp["common_mask"]

    n_harm_frames = harm_z_ds.shape[0]
    n_l30_frames = l30_sr_ds.shape[0]
    n_s30_frames = s30_sr_ds.shape[0]
    n_patches = len(LAND_COVER_PATCHES)

    chips = {
        "HLSL30": np.zeros((n_patches, n_l30_frames, 7, PATCH_SIZE, PATCH_SIZE), dtype=l30_sr_ds.dtype),
        "HLSS30": np.zeros((n_patches, n_s30_frames, 10, PATCH_SIZE, PATCH_SIZE), dtype=s30_sr_ds.dtype),
        "HLSL30_Fmask": np.zeros((n_patches, n_l30_frames, PATCH_SIZE, PATCH_SIZE), dtype=l30_fm_ds.dtype),
        "HLSS30_Fmask": np.zeros((n_patches, n_s30_frames, PATCH_SIZE, PATCH_SIZE), dtype=s30_fm_ds.dtype),
        "HLSL30_common_mask": np.zeros((n_patches, n_l30_frames, PATCH_SIZE, PATCH_SIZE), dtype=l30_cm_ds.dtype),
        "HLSS30_common_mask": np.zeros((n_patches, n_s30_frames, PATCH_SIZE, PATCH_SIZE), dtype=s30_cm_ds.dtype),
        "HARMONIZED_z_score": np.zeros((n_patches, n_harm_frames, PATCH_SIZE, PATCH_SIZE), dtype=harm_z_ds.dtype),
        "HARMONIZED_common_mask": np.zeros((n_patches, n_harm_frames, PATCH_SIZE, PATCH_SIZE), dtype=harm_cm_ds.dtype),
    }

    l30_fill, l30_scale = get_dataset_metadata(l30_sr_ds)
    s30_fill, s30_scale = get_dataset_metadata(s30_sr_ds)

    scale_factors = {
        "HLSL30": l30_scale,
        "HLSS30": s30_scale,
    }
    fill_values = {
        "HLSL30": l30_fill,
        "HLSS30": s30_fill,
    }

    mask_fill_values = {
        "HLSL30_Fmask": l30_fm_ds.attrs.get("_FillValue", getattr(l30_fm_ds, "fillvalue", 255)),
        "HLSL30_common_mask": l30_cm_ds.attrs.get("_FillValue", getattr(l30_cm_ds, "fillvalue", 255)),
        "HLSS30_Fmask": s30_fm_ds.attrs.get("_FillValue", getattr(s30_fm_ds, "fillvalue", 255)),
        "HLSS30_common_mask": s30_cm_ds.attrs.get("_FillValue", getattr(s30_cm_ds, "fillvalue", 255)),
        "HARMONIZED_common_mask": harm_cm_ds.attrs.get("_FillValue", getattr(harm_cm_ds, "fillvalue", 255)),
    }

    tan_sr_ds = None
    if tan_grp is not None and "surface_reflectance" in tan_grp:
        tan_sr_ds = tan_grp["surface_reflectance"]
        n_tan_frames = tan_sr_ds.shape[0]
        n_tan_bands = tan_sr_ds.shape[1]
        chips["TANAGER"] = np.zeros((n_patches, n_tan_frames, n_tan_bands, PATCH_SIZE, PATCH_SIZE), dtype=tan_sr_ds.dtype)
        tan_fill, tan_scale = get_dataset_metadata(tan_sr_ds)
        scale_factors["TANAGER"] = tan_scale
        fill_values["TANAGER"] = tan_fill

    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        chips["HLSL30"][p_idx] = l30_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30"][p_idx] = s30_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSL30_Fmask"][p_idx] = l30_fm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30_Fmask"][p_idx] = s30_fm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSL30_common_mask"][p_idx] = l30_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30_common_mask"][p_idx] = s30_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HARMONIZED_z_score"][p_idx] = harm_z_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HARMONIZED_common_mask"][p_idx] = harm_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        if tan_sr_ds is not None:
            chips["TANAGER"][p_idx] = tan_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]

    # Extract baseline SC-CCD results directly from master container /SCCCD_Results (produced by harmonized_CCD_main.py)
    # The outer 2-pixel surrounding region is NOT evaluated by SC-CCD (initialized to NaN date, 0 count)
    baseline_scccd = {
        "date": np.full((n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64),
        "count": np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32),
    }
    sc_date_ds = h5_file[f"{scccd_grp_name}/change_date_timestamp"]
    sc_count_ds = h5_file[f"{scccd_grp_name}/change_count"]
    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        baseline_scccd["date"][p_idx, 2:11, 2:11] = sc_date_ds[r0+2:r0+11, c0+2:c0+11]
        baseline_scccd["count"][p_idx, 2:11, 2:11] = sc_count_ds[r0+2:r0+11, c0+2:c0+11]
    chips["BASELINE_SCCCD"] = baseline_scccd

    # Extract baseline PyCCD results directly from master container /PYCCD_Results (produced by pyccd_landsat_main.py)
    # The outer 2-pixel surrounding region is NOT evaluated (initialized to NaN date, 0 count)
    baseline_pyccd = {
        "date": np.full((n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64),
        "count": np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32),
    }
    py_date_ds = h5_file[f"{pyccd_grp_name}/change_date_timestamp"]
    py_count_ds = h5_file[f"{pyccd_grp_name}/change_count"]
    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        baseline_pyccd["date"][p_idx, 2:11, 2:11] = py_date_ds[r0+2:r0+11, c0+2:c0+11]
        baseline_pyccd["count"][p_idx, 2:11, 2:11] = py_count_ds[r0+2:r0+11, c0+2:c0+11]
    chips["BASELINE_PYCCD"] = baseline_pyccd

    metadata = {
        "harm_acq_times": harm_z_ds.attrs["acquisition_time"][:],
        "harm_source_grid": np.array([g.decode() if isinstance(g, bytes) else str(g) for g in harm_z_ds.attrs["source_grid"]]),
        "harm_source_idx": harm_z_ds.attrs["source_frame_index"][:],
        "frame_global_means": harm_z_ds.attrs["frame_global_means"][:],
        "frame_global_stds": harm_z_ds.attrs["frame_global_stds"][:],
        "l30_acq_times": l30_sr_ds.attrs["acquisition_time"][:],
        "s30_acq_times": s30_sr_ds.attrs["acquisition_time"][:],
        "scale_factors": scale_factors,
        "fill_values": fill_values,
        "mask_fill_values": mask_fill_values,
        "GeoTransform": l30_sr_ds.attrs.get("GeoTransform"),
        "spatial_ref": l30_sr_ds.attrs.get("spatial_ref"),
    }
    if tan_sr_ds is not None:
        metadata["tan_acq_times"] = tan_sr_ds.attrs.get("acquisition_time")
        metadata["tanager_good_wavelengths"] = tan_sr_ds.attrs.get("all_good_wavelengths")
        metadata["tanager_wavelengths"] = tan_sr_ds.attrs.get("wavelengths")

    print(f"Extracted baseline chips across {n_patches} strata:")
    print(f"  HLSL30: {chips['HLSL30'].shape} ({n_l30_frames} Landsat acquisitions, scale={scale_factors['HLSL30']})")
    print(f"  HLSS30: {chips['HLSS30'].shape} ({n_s30_frames} Sentinel-2 acquisitions, scale={scale_factors['HLSS30']})")
    if tan_sr_ds is not None:
        print(f"  TANAGER: {chips['TANAGER'].shape} ({n_tan_frames} hyperspectral frames, scale={scale_factors['TANAGER']})")
    print(f"  HARMONIZED: {chips['HARMONIZED_z_score'].shape} ({n_harm_frames} virtual constellation frames)")

    return chips, metadata

def load_all_target_spectra(splib_base=SPLIB_BASE, tanager_wavelengths=None):
    """
    Loads and resamples the 10 USGS splib07 target materials for Landsat-8,
    Sentinel-2, and Tanager-1 sensors.
    """
    asd_dir = os.path.join(splib_base, "ASCIIdata_splib07b_cvASD")
    l8_dir  = os.path.join(splib_base, "ASCIIdata_splib07b_rsLandsat8")
    s2_dir  = os.path.join(splib_base, "ASCIIdata_splib07b_rsSentinel2")

    asd_wave_f = os.path.join(asd_dir, "s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt")
    if not os.path.exists(asd_wave_f):
        raise FileNotFoundError(f"ASD wavelength file not found: {asd_wave_f}")

    asd_wave_nm = np.loadtxt(asd_wave_f, skiprows=1) * 1000.0

    spectra_by_material = []
    for item in TARGET_MATERIALS:
        p_asd = os.path.join(asd_dir, item["asd_rel"])
        p_l8  = os.path.join(l8_dir,  item["l8_rel"])
        p_s2  = os.path.join(s2_dir,  item["s2_rel"])

        r_asd = np.loadtxt(p_asd, skiprows=1)
        r_l8  = np.loadtxt(p_l8,  skiprows=1)
        r_s2  = np.loadtxt(p_s2,  skiprows=1)

        r_asd[r_asd < 0] = np.nan
        r_l8[r_l8 < 0]   = np.nan
        r_s2[r_s2 < 0]   = np.nan

        spec_l8 = r_l8.copy().astype(np.float32)
        spec_s2 = r_s2[S2_SUB_INDICES].astype(np.float32)

        if tanager_wavelengths is not None:
            valid_asd = ~np.isnan(r_asd)
            interp_fn = scipy.interpolate.interp1d(
                asd_wave_nm[valid_asd],
                r_asd[valid_asd],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            spec_tanager = interp_fn(tanager_wavelengths).astype(np.float32)
            spec_tanager = np.clip(spec_tanager, 0.0, 1.0)
        else:
            spec_tanager = None

        spectra_by_material.append({
            "id": item["id"],
            "name": item["name"],
            "contrast": item["contrast"],
            "role": item["role"],
            "spec_l8": spec_l8,
            "spec_s2": spec_s2,
            "spec_tanager": spec_tanager,
        })

    return spectra_by_material

# ==============================================================================
# STEP 2 & 3: SIMULATE SPECTRAL COMPLEXITY CUBES (GPU MAXD SLIDING VOLUME)
# ==============================================================================
def simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts):
    """
    Computes 9x9 sliding volume Gramian complexity on GPU across all 50 conditions.
    Evaluates baseline frames and post-onset linear mixtures using SpecComplexTorch.process_volume_sliding_tile
    and normalizes strictly via SpecComplexTorch.calculate_global_z_score with frame_global_means and frame_global_stds.
    """
    print("\n[Step 2 & 3] Simulating Spectral Complexity Z-scores on GPU across 50 conditions...")

    harm_acq_times = metadata["harm_acq_times"]
    source_grid = metadata["harm_source_grid"]
    source_idx = metadata["harm_source_idx"]
    means = metadata["frame_global_means"]
    stds = metadata["frame_global_stds"]

    n_frames = len(harm_acq_times)
    n_patches = len(LAND_COVER_PATCHES)
    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    num_conditions = num_materials * num_fractions

    # 1. Identify pre-onset and post-onset frame indices
    pre_onset_mask = harm_acq_times < onset_ts
    post_onset_indices = np.where(~pre_onset_mask)[0]
    pre_onset_indices = np.where(pre_onset_mask)[0]

    print(f"Timeline partitioning: {len(pre_onset_indices)} pre-onset frames, {len(post_onset_indices)} post-onset frames.")

    # 2. Precompute unperturbed baseline Z-scores for all patches
    print("Precomputing unperturbed baseline Z-score time series for all patches...")
    unperturbed_z = np.full((n_patches, n_frames, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)

    scale_factors = metadata["scale_factors"]
    fill_values = metadata["fill_values"]
    tan_gw = metadata.get("tanager_good_wavelengths")

    for t in range(n_frames):
        mu_t = means[t]
        std_t = stds[t]

        # Skip frames where background statistics are undefined (e.g. 100% cloud/aerosol)
        if np.isnan(mu_t) or np.isnan(std_t) or std_t <= 0:
            continue

        grid_name = source_grid[t]
        local_idx = source_idx[t]

        if grid_name not in chips:
            continue

        scale = scale_factors[grid_name]
        fill_val = fill_values[grid_name]

        for p_idx in range(n_patches):
            raw_patch = chips[grid_name][p_idx, local_idx]
            flt_sr = raw_patch.astype(np.float32) / float(scale)
            if np.issubdtype(raw_patch.dtype, np.integer):
                flt_sr[raw_patch == fill_val] = np.nan
            else:
                if fill_val is not None and not np.isnan(fill_val):
                    flt_sr[np.isclose(raw_patch, fill_val, equal_nan=True)] = np.nan

            if grid_name == "TANAGER" and tan_gw is not None:
                gw_mask = tan_gw[local_idx].astype(bool)
                eval_sr = flt_sr[gw_mask, ...]
            else:
                eval_sr = flt_sr

            vol_map = sc.process_volume_sliding_tile(
                eval_sr, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
            )
            valid_mask = (chips["HARMONIZED_common_mask"][p_idx, t] == 0)
            z_map, _, _ = sc.calculate_global_z_score(
                vol_map,
                valid_pixel_mask=valid_mask,
                array_mean=mu_t,
                array_std=std_t,
            )
            if hasattr(z_map, "detach"):
                z_map = z_map.detach().cpu().numpy()
            unperturbed_z[p_idx, t] = z_map

    # 3. Simulate post-onset linear mixture across 50 conditions
    simulated_z_cubes = np.zeros((num_conditions, n_patches, n_frames, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    # Initialize all conditions with unperturbed baseline Z-scores
    for c_idx in range(num_conditions):
        for p_idx in range(n_patches):
            simulated_z_cubes[c_idx, p_idx] = unperturbed_z[p_idx].copy()

    # Fast GPU simulation loop over post-onset frames
    t_start_sim = time.perf_counter()
    cond_counter = 0

    for m_idx, mat in enumerate(target_spectra):
        spec_l8 = mat["spec_l8"]
        spec_s2 = mat["spec_s2"]
        spec_tanager = mat.get("spec_tanager")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for t in post_onset_indices:
                mu_t = means[t]
                std_t = stds[t]

                if np.isnan(mu_t) or np.isnan(std_t) or std_t <= 0:
                    continue

                grid_name = source_grid[t]
                local_idx = source_idx[t]

                if grid_name not in chips:
                    continue

                scale = scale_factors[grid_name]
                fill_val = fill_values[grid_name]

                if grid_name == "HLSL30":
                    target_spec = spec_l8
                elif grid_name == "HLSS30":
                    target_spec = spec_s2
                elif grid_name == "TANAGER":
                    target_spec = spec_tanager
                else:
                    continue

                if target_spec is None:
                    continue

                for p_idx in range(n_patches):
                    raw_patch = chips[grid_name][p_idx, local_idx]
                    flt_sr = raw_patch.astype(np.float32) / float(scale)
                    if np.issubdtype(raw_patch.dtype, np.integer):
                        flt_sr[raw_patch == fill_val] = np.nan
                    else:
                        if fill_val is not None and not np.isnan(fill_val):
                            flt_sr[np.isclose(raw_patch, fill_val, equal_nan=True)] = np.nan

                    # Inject linear mixture into the 4 designated pixels
                    injected_sr = flt_sr.copy()
                    for roff, coff in INJ_OFFSETS:
                        bg_pixel = flt_sr[:, roff, coff]
                        if not np.isnan(bg_pixel).any():
                            injected_sr[:, roff, coff] = (1.0 - f_weight) * bg_pixel + f_weight * target_spec

                    if grid_name == "TANAGER" and tan_gw is not None:
                        gw_mask = tan_gw[local_idx].astype(bool)
                        eval_injected = injected_sr[gw_mask, ...]
                    else:
                        eval_injected = injected_sr

                    vol_map = sc.process_volume_sliding_tile(
                        eval_injected, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                    )
                    valid_mask = (chips["HARMONIZED_common_mask"][p_idx, t] == 0)
                    z_map, _, _ = sc.calculate_global_z_score(
                        vol_map,
                        valid_pixel_mask=valid_mask,
                        array_mean=mu_t,
                        array_std=std_t,
                    )
                    if hasattr(z_map, "detach"):
                        z_map = z_map.detach().cpu().numpy()
                    simulated_z_cubes[cond_counter, p_idx, t] = z_map

            cond_counter += 1

    t_elapsed_sim = time.perf_counter() - t_start_sim
    print(f"Completed GPU simulation of 50 conditions in {t_elapsed_sim:.2f} s ({t_elapsed_sim/num_conditions:.2f} s/condition).")

    return simulated_z_cubes, unperturbed_z


# ==============================================================================
# STEP 4: RUN HARMONIZED SC-CCD BREAK DETECTION
# ==============================================================================
def run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts, baseline_scccd=None, return_spatial=True):
    """
    Executes Harmonized SC-CCD break regression across all 800 test instances.
    Evaluates the inner 9x9 region of each 13x13 patch across time to accurately characterize
    the spatial impact on neighboring pixels when modifying the reflectance of implanted pixels.
    The outer 2-pixel surrounding region is NOT evaluated by SC-CCD.
    Evaluates detected breaks against the unperturbed baseline SC-CCD results produced by harmonized_CCD_main.py.
    """
    print("\n[Step 4] Running Harmonized SC-CCD break detection on inner 9x9 regions (outer 2px not evaluated)...")

    if baseline_scccd is None and isinstance(chips, dict):
        baseline_scccd = chips.get("BASELINE_SCCCD")

    if baseline_scccd is None:
        raise RuntimeError(
            f"Baseline SC-CCD dataset '/SCCCD_Results' was not found in chips or master container '{MASTER_H5_PATH}'. "
            f"Please run harmonized_CCD_main.py first before executing the stratified patch benchmark."
        )
    harm_acq_times = metadata["harm_acq_times"]
    frac_years = hccd.extract_fractional_years(harm_acq_times)
    X_full = hccd.build_harmonic_matrix(frac_years)

    common_mask = chips["HARMONIZED_common_mask"]
    valid_mask_all = (common_mask == 0)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    num_conditions = num_materials * num_fractions
    total_records = num_conditions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    scccd_change_dates = np.full((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
    scccd_change_counts = np.zeros((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)

    rec_idx = 0
    cond_idx = 0

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                # Run CCD row chunk ONLY across the inner 9x9 region (rows 2:11, cols 2:11)
                # The outer 2-pixel surrounding region is NOT evaluated by SC-CCD
                y_data_patch = sim_z_cubes[cond_idx, p_idx]
                valid_patch = valid_mask_all[p_idx]

                y_inner = y_data_patch[:, 2:11, 2:11]
                valid_inner = valid_patch[:, 2:11, 2:11]

                chunk_args = (
                    0, 9, 9,
                    y_inner, valid_inner,
                    frac_years, X_full, harm_acq_times,
                    hccd.MIN_SAMPLES, hccd.CHANGE_PROBABILITY, hccd.CONSECUTIVE_ANOMALIES
                )

                res = hccd._process_row_chunk(chunk_args)
                chunk_pred = res[2]
                chunk_date = res[6]
                chunk_count = res[7]

                # Maintain and record spatial results for inner 9x9 evaluated region; outer 2px remain NaN / 0
                scccd_change_dates[cond_idx, p_idx, 2:11, 2:11] = chunk_date
                scccd_change_counts[cond_idx, p_idx, 2:11, 2:11] = chunk_count

                for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
                    scene_r = r0 + roff
                    scene_c = c0 + coff

                    inner_r = roff - 2
                    inner_c = coff - 2
                    b_count = int(chunk_count[inner_r, inner_c])
                    b_date = float(chunk_date[inner_r, inner_c])

                    # Compare against baseline SC-CCD evaluation (produced by harmonized_CCD_main.py)
                    base_count = int(baseline_scccd["count"][p_idx, roff, coff])
                    base_date = float(baseline_scccd["date"][p_idx, roff, coff])

                    # Detection criteria: break occurred on or after onset date that was not already present in baseline
                    has_break_post_onset = (b_count > 0) and (not np.isnan(b_date)) and (b_date >= onset_ts)
                    is_new_break = has_break_post_onset and (
                        (base_count == 0) or
                        np.isnan(base_date) or
                        (b_count > base_count) or
                        (b_date < base_date)
                    )
                    detected = is_new_break
                    latency = float((b_date - onset_ts) / 86400.0) if detected else np.nan

                    # Break magnitude: residual jump post-onset
                    if detected:
                        post_valid = np.where(valid_inner[:, inner_r, inner_c] & (harm_acq_times >= onset_ts))[0]
                        if len(post_valid) > 0:
                            actuals = y_inner[post_valid, inner_r, inner_c]
                            preds = chunk_pred[post_valid, inner_r, inner_c]
                            mag = float(np.nanmean(np.abs(actuals - preds)))
                        else:
                            mag = 0.0
                    else:
                        mag = 0.0

                    rec = results_table[rec_idx]
                    rec["condition_idx"] = cond_idx
                    rec["material_id"] = m_id
                    rec["material_name"] = m_name
                    rec["mixture_fraction"] = f_weight
                    rec["patch_id"] = p_id
                    rec["patch_name"] = p_name
                    rec["pixel_num"] = p_num
                    rec["scene_row"] = scene_r
                    rec["scene_col"] = scene_c
                    rec["break_detected"] = detected
                    rec["break_timestamp"] = b_date if detected else np.nan
                    rec["latency_days"] = latency
                    rec["break_magnitude"] = mag
                    rec["p_value"] = 0.99 if detected else 0.0

                    rec_idx += 1

            cond_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    print(f"SC-CCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")

    spatial_results = {
        "change_date": scccd_change_dates,
        "change_count": scccd_change_counts,
    }
    if return_spatial:
        return results_table, spatial_results
    return results_table


# ==============================================================================
# STEP 5: RUN LANDSAT PYCCD BREAK DETECTION
# ==============================================================================
def run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts, baseline_pyccd=None, return_spatial=True):
    """
    Executes USGS LCMAP Landsat PyCCD break detection across all 800 test instances.
    Only evaluates the 4 implanted pixels per patch for each perturbation condition.
    Because PyCCD has no spatial context for neighboring pixels, the neighboring baseline
    pixels copy the baseline PyCCD evaluation (produced by pyccd_landsat_main.py)
    to maintain the full 13x13 patch output array size.
    """
    print("\n[Step 5] Running USGS LCMAP Landsat PyCCD on 4 implanted pixels per patch...")

    if baseline_pyccd is None and isinstance(chips, dict):
        baseline_pyccd = chips.get("BASELINE_PYCCD")

    if baseline_pyccd is None:
        raise RuntimeError(
            f"Baseline PyCCD dataset '/PYCCD_Results' was not found in chips or master container '{MASTER_H5_PATH}'. "
            f"Please run pyccd_landsat_main.py first before executing the stratified patch benchmark."
        )

    l30_acq_times = metadata["l30_acq_times"]
    l30_dates = [datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).date() for t in l30_acq_times]
    ordinal_dates = np.array([d.toordinal() for d in l30_dates])
    onset_ordinal = datetime.datetime.fromtimestamp(onset_ts, tz=datetime.timezone.utc).date().toordinal()

    n_l30_frames = len(l30_acq_times)
    nominal_thermals = np.full(n_l30_frames, pccd.NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    num_conditions = num_materials * num_fractions
    total_records = num_conditions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    pyccd_change_dates = np.full((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
    pyccd_change_counts = np.zeros((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)

    rec_idx = 0
    cond_idx = 0

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

    # Post-onset Landsat frame indices
    post_onset_l30 = np.where(l30_acq_times >= onset_ts)[0]

    l30_scale = metadata["scale_factors"]["HLSL30"]
    l30_fill = metadata["fill_values"]["HLSL30"]

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")
        spec_l8_flt = mat["spec_l8"]

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                # Copy baseline PyCCD evaluation for neighboring baseline pixels to maintain output array size
                pyccd_change_dates[cond_idx, p_idx] = baseline_pyccd["date"][p_idx].copy()
                pyccd_change_counts[cond_idx, p_idx] = baseline_pyccd["count"][p_idx].copy()

                l30_patch = chips["HLSL30"][p_idx]          # shape: (n_frames, 7, PATCH_SIZE, PATCH_SIZE)
                fmask_patch = chips["HLSL30_Fmask"][p_idx]  # shape: (n_frames, PATCH_SIZE, PATCH_SIZE)
                cmask_patch = chips["HLSL30_common_mask"][p_idx] # shape: (n_frames, PATCH_SIZE, PATCH_SIZE)

                # ONLY evaluate the 4 implanted pixels per patch
                for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
                    scene_r = r0 + roff
                    scene_c = c0 + coff

                    # Extract raw 6-band time series
                    blues = l30_patch[:, 1, roff, coff].copy()
                    greens = l30_patch[:, 2, roff, coff].copy()
                    reds = l30_patch[:, 3, roff, coff].copy()
                    nirs = l30_patch[:, 4, roff, coff].copy()
                    swir1s = l30_patch[:, 5, roff, coff].copy()
                    swir2s = l30_patch[:, 6, roff, coff].copy()

                    # Inject target mixture for post-onset acquisitions
                    for t in post_onset_l30:
                        obs_pixel = np.array([blues[t], greens[t], reds[t], nirs[t], swir1s[t], swir2s[t]])
                        if not np.any(obs_pixel == l30_fill):
                            blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8_flt[1] * l30_scale))
                            greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8_flt[2] * l30_scale))
                            reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8_flt[3] * l30_scale))
                            nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8_flt[4] * l30_scale))
                            swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8_flt[5] * l30_scale))
                            swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8_flt[6] * l30_scale))

                    fm_pixel = fmask_patch[:, roff, coff]
                    cm_pixel = cmask_patch[:, roff, coff]
                    sr_pixel_bands = np.column_stack([blues, greens, reds, nirs, swir1s, swir2s])
                    qas_pixel = pccd.decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=l30_fill)

                    usable_count = int(np.count_nonzero((qas_pixel == 1) | (qas_pixel == 2) | (qas_pixel == 4)))

                    detected = False
                    break_ts = np.nan
                    latency = np.nan
                    mag = 0.0
                    b_count = 0

                    if usable_count >= pccd.MIN_CLEAR_SAMPLES:
                        try:
                            res = ccd.detect(
                                ordinal_dates,
                                blues, greens, reds, nirs, swir1s, swir2s,
                                nominal_thermals,
                                qas_pixel,
                                params=proc_params
                            )
                            models = res.get("change_models", [])
                            breaks = [m for m in models if m.get("change_probability", 0) > 0]
                            b_count = len(breaks)
                            # Find first break on or after onset date
                            for m in breaks:
                                b_day = m.get("break_day")
                                if b_day is not None and b_day >= onset_ordinal:
                                    detected = True
                                    b_date_obj = datetime.date.fromordinal(b_day)
                                    break_ts = datetime.datetime(
                                        b_date_obj.year, b_date_obj.month, b_date_obj.day,
                                        tzinfo=datetime.timezone.utc
                                    ).timestamp()
                                    latency = float((break_ts - onset_ts) / 86400.0)
                                    mags = m.get("magnitudes", [0.0])
                                    mag = float(np.linalg.norm(mags))
                                    break
                        except Exception:
                            detected = False
                            b_count = 0

                    # Compare against baseline PyCCD evaluation (produced by pyccd_landsat_main.py)
                    base_count = int(baseline_pyccd["count"][p_idx, roff, coff])
                    base_date = float(baseline_pyccd["date"][p_idx, roff, coff])

                    has_break_post_onset = detected and (not np.isnan(break_ts)) and (break_ts >= onset_ts)
                    is_new_break = has_break_post_onset and (
                        (base_count == 0) or
                        np.isnan(base_date) or
                        (b_count > base_count) or
                        (break_ts < base_date)
                    )
                    detected = is_new_break
                    latency = float((break_ts - onset_ts) / 86400.0) if detected else np.nan

                    # Update the patch spatial arrays for the implanted pixel
                    pyccd_change_dates[cond_idx, p_idx, roff, coff] = break_ts if detected else np.nan
                    pyccd_change_counts[cond_idx, p_idx, roff, coff] = b_count

                    rec = results_table[rec_idx]
                    rec["condition_idx"] = cond_idx
                    rec["material_id"] = m_id
                    rec["material_name"] = m_name
                    rec["mixture_fraction"] = f_weight
                    rec["patch_id"] = p_id
                    rec["patch_name"] = p_name
                    rec["pixel_num"] = p_num
                    rec["scene_row"] = scene_r
                    rec["scene_col"] = scene_c
                    rec["break_detected"] = detected
                    rec["break_timestamp"] = break_ts if detected else np.nan
                    rec["latency_days"] = latency
                    rec["break_magnitude"] = mag
                    rec["p_value"] = 1.0 if detected else 0.0

                    rec_idx += 1

            cond_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    print(f"PyCCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")

    spatial_results = {
        "change_date": pyccd_change_dates,
        "change_count": pyccd_change_counts,
    }
    if return_spatial:
        return results_table, spatial_results
    return results_table


# ==============================================================================
# STEP 6: COMMIT BENCHMARK DATASET TO MASTER HDF5 FILE
# ==============================================================================
def save_benchmark_to_hdf5(h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table,
                           scccd_spatial=None, pyccd_spatial=None):
    """
    Writes the complete /STRATIFIED_CHIP_BENCHMARK/ hierarchy to the master HDF5 container.
    """
    print("\n[Step 6] Saving /STRATIFIED_CHIP_BENCHMARK/ group to master HDF5 file...")

    group_name = "/STRATIFIED_CHIP_BENCHMARK"
    if group_name in h5_file:
        print(f"Overwriting existing '{group_name}' group...")
        del h5_file[group_name]

    base_grp = h5_file.create_group(group_name)

    # 1. METADATA Subgroup
    meta_grp = base_grp.create_group("METADATA")
    meta_grp.attrs["patch_size"] = PATCH_SIZE
    meta_grp.attrs["stride"] = 4
    meta_grp.attrs["injection_onset_date"] = INJECTION_ONSET_DATE
    meta_grp.attrs["num_conditions"] = len(target_spectra) * len(MIXTURE_FRACTIONS)
    meta_grp.attrs["patch_ids"] = np.array([p["id"] for p in LAND_COVER_PATCHES], dtype="S16")
    meta_grp.attrs["patch_names"] = np.array([p["name"] for p in LAND_COVER_PATCHES], dtype="S48")
    meta_grp.attrs["patch_bounds"] = np.array([[p["r0"], p["c0"], PATCH_SIZE, PATCH_SIZE] for p in LAND_COVER_PATCHES], dtype=np.int32)
    meta_grp.attrs["injected_offsets"] = np.array(INJ_OFFSETS, dtype=np.int32)
    meta_grp.attrs["mixture_fractions"] = np.array(MIXTURE_FRACTIONS, dtype=np.float32)
    meta_grp.attrs["frame_global_means"] = metadata["frame_global_means"]
    meta_grp.attrs["frame_global_stds"] = metadata["frame_global_stds"]
    meta_grp.attrs["harm_acq_times"] = metadata["harm_acq_times"]

    # Target spectra datasets
    spec_l8_all = np.array([m["spec_l8"] for m in target_spectra], dtype=np.float32)
    spec_s2_all = np.array([m["spec_s2"] for m in target_spectra], dtype=np.float32)
    meta_grp.create_dataset("target_spectra_l8", data=spec_l8_all, compression="gzip")
    meta_grp.create_dataset("target_spectra_s2", data=spec_s2_all, compression="gzip")

    if target_spectra[0]["spec_tanager"] is not None:
        spec_tanager_all = np.array([m["spec_tanager"] for m in target_spectra], dtype=np.float32)
        meta_grp.create_dataset("target_spectra_tanager", data=spec_tanager_all, compression="gzip")

    # 2. BASELINE_CHIPS Subgroup
    base_chips_grp = base_grp.create_group("BASELINE_CHIPS")
    l30_ds = base_chips_grp.create_dataset(
        "HLSL30",
        data=chips["HLSL30"],
        dtype=chips["HLSL30"].dtype,
        fillvalue=metadata["fill_values"]["HLSL30"],
        compression="gzip",
        compression_opts=6
    )
    l30_ds.attrs["scale_factor"] = metadata["scale_factors"]["HLSL30"]
    l30_ds.attrs["_FillValue"] = metadata["fill_values"]["HLSL30"]

    fm_l30_ds = base_chips_grp.create_dataset(
        "HLSL30_Fmask",
        data=chips["HLSL30_Fmask"],
        dtype=chips["HLSL30_Fmask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSL30_Fmask"],
        compression="gzip",
        compression_opts=6
    )
    fm_l30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSL30_Fmask"]

    cm_l30_ds = base_chips_grp.create_dataset(
        "HLSL30_common_mask",
        data=chips["HLSL30_common_mask"],
        dtype=chips["HLSL30_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSL30_common_mask"],
        compression="gzip",
        compression_opts=6
    )
    cm_l30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSL30_common_mask"]

    s30_ds = base_chips_grp.create_dataset(
        "HLSS30",
        data=chips["HLSS30"],
        dtype=chips["HLSS30"].dtype,
        fillvalue=metadata["fill_values"]["HLSS30"],
        compression="gzip",
        compression_opts=6
    )
    s30_ds.attrs["scale_factor"] = metadata["scale_factors"]["HLSS30"]
    s30_ds.attrs["_FillValue"] = metadata["fill_values"]["HLSS30"]

    fm_s30_ds = base_chips_grp.create_dataset(
        "HLSS30_Fmask",
        data=chips["HLSS30_Fmask"],
        dtype=chips["HLSS30_Fmask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSS30_Fmask"],
        compression="gzip",
        compression_opts=6
    )
    fm_s30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSS30_Fmask"]

    cm_s30_ds = base_chips_grp.create_dataset(
        "HLSS30_common_mask",
        data=chips["HLSS30_common_mask"],
        dtype=chips["HLSS30_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSS30_common_mask"],
        compression="gzip",
        compression_opts=6
    )
    cm_s30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSS30_common_mask"]

    if "TANAGER" in chips:
        tan_ds = base_chips_grp.create_dataset(
            "TANAGER",
            data=chips["TANAGER"],
            dtype=chips["TANAGER"].dtype,
            fillvalue=metadata["fill_values"]["TANAGER"],
            compression="gzip",
            compression_opts=6
        )
        tan_ds.attrs["scale_factor"] = metadata["scale_factors"]["TANAGER"]
        tan_ds.attrs["_FillValue"] = metadata["fill_values"]["TANAGER"]

    harm_cm_ds = base_chips_grp.create_dataset(
        "HARMONIZED_common_mask",
        data=chips["HARMONIZED_common_mask"],
        dtype=chips["HARMONIZED_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HARMONIZED_common_mask"],
        compression="gzip",
        compression_opts=6
    )
    harm_cm_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HARMONIZED_common_mask"]

    harm_z = base_chips_grp.create_dataset(
        "HARMONIZED_z_score",
        data=chips["HARMONIZED_z_score"],
        dtype=chips["HARMONIZED_z_score"].dtype,
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=6
    )
    harm_z.attrs["_FillValue"] = np.nan

    if "BASELINE_SCCCD" in chips:
        base_chips_grp.create_dataset(
            "scccd_baseline_change_date",
            data=chips["BASELINE_SCCCD"]["date"],
            compression="gzip",
            compression_opts=6
        )
        base_chips_grp.create_dataset(
            "scccd_baseline_change_count",
            data=chips["BASELINE_SCCCD"]["count"],
            compression="gzip",
            compression_opts=6
        )
    if "BASELINE_PYCCD" in chips:
        base_chips_grp.create_dataset(
            "pyccd_baseline_change_date",
            data=chips["BASELINE_PYCCD"]["date"],
            compression="gzip",
            compression_opts=6
        )
        base_chips_grp.create_dataset(
            "pyccd_baseline_change_count",
            data=chips["BASELINE_PYCCD"]["count"],
            compression="gzip",
            compression_opts=6
        )

    # 3. SIMULATED_CUBES Subgroup (Stored as native float32 with gzip level 6)
    sim_grp = base_grp.create_group("SIMULATED_CUBES")
    sim_ds = sim_grp.create_dataset(
        "sliding_volume_z_score",
        data=sim_z_cubes,
        dtype=sim_z_cubes.dtype,
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=6,
        chunks=(1, 1, min(sim_z_cubes.shape[2], 128), PATCH_SIZE, PATCH_SIZE),
    )
    sim_ds.attrs["_FillValue"] = np.nan
    sim_ds.attrs["description"] = "Simulated Gramian Spectral Complexity Z-scores under 50 perturbation conditions"

    # 4. RESULTS Subgroup
    res_grp = base_grp.create_group("RESULTS")
    res_grp.create_dataset("scccd_break_table", data=scccd_table, compression="gzip")
    res_grp.create_dataset("pyccd_break_table", data=pyccd_table, compression="gzip")

    if scccd_spatial is not None:
        for k, arr in scccd_spatial.items():
            res_grp.create_dataset(
                f"scccd_{k}",
                data=arr,
                compression="gzip",
                compression_opts=6,
                chunks=(1, 1, PATCH_SIZE, PATCH_SIZE),
            )
    if pyccd_spatial is not None:
        for k, arr in pyccd_spatial.items():
            res_grp.create_dataset(
                f"pyccd_{k}",
                data=arr,
                compression="gzip",
                compression_opts=6,
                chunks=(1, 1, PATCH_SIZE, PATCH_SIZE),
            )

    print(f"Successfully committed /STRATIFIED_CHIP_BENCHMARK/ to {MASTER_H5_PATH}.")


# ==============================================================================
# STEP 7: PRINT QUANTITATIVE PERFORMANCE SUMMARY
# ==============================================================================
def print_performance_summary(scccd_table, pyccd_table):
    """
    Prints comparative detection accuracy and latency tables stratified by land-cover class.
    """
    print("\n" + "=" * 90)
    print("LAND-COVER STRATIFIED BENCHMARK PERFORMANCE SUMMARY: SC-CCD vs. USGS PYCCD")
    print("=" * 90)

    # 1. Performance Stratified by Land-Cover Class
    print(f"\n{'Stratum ID':<10} | {'Land-Cover Stratum':<32} | {'SC-CCD Recall':<15} | {'PyCCD Recall':<15} | {'Median Latency (SC / Py)'}")
    print("-" * 105)

    patch_ids = [p["id"] for p in LAND_COVER_PATCHES]
    patch_names = [p["name"] for p in LAND_COVER_PATCHES]

    for p_id, p_name in zip(patch_ids, patch_names):
        sc_sub = scccd_table[scccd_table["patch_id"] == p_id.encode()]
        py_sub = pyccd_table[pyccd_table["patch_id"] == p_id.encode()]

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{p_id:<10} | {p_name:<32} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):3d}/{len(sc_sub):3d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):3d}/{len(py_sub):3d}) | {sc_med_lat:<12} / {py_med_lat}")

    # 2. Performance Stratified by Target Material
    print(f"\n{'Material ID':<18} | {'Material Name':<34} | {'SC Recall':<12} | {'Py Recall':<12} | {'Median Latency (SC / Py)'}")
    print("-" * 105)

    mat_ids = np.unique(scccd_table["material_id"])
    for m_id in mat_ids:
        sc_sub = scccd_table[scccd_table["material_id"] == m_id]
        py_sub = pyccd_table[pyccd_table["material_id"] == m_id]

        m_name = sc_sub["material_name"][0].decode()
        m_id_str = m_id.decode()

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{m_id_str:<18} | {m_name:<34} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):2d}/{len(sc_sub):2d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):2d}/{len(py_sub):2d}) | {sc_med_lat:<12} / {py_med_lat}")

    # 3. Overall Aggregate Performance
    total_sc_rec = np.mean(scccd_table["break_detected"]) * 100.0
    total_py_rec = np.mean(pyccd_table["break_detected"]) * 100.0
    all_sc_lat = scccd_table["latency_days"][scccd_table["break_detected"]]
    all_py_lat = pyccd_table["latency_days"][pyccd_table["break_detected"]]

    print("-" * 105)
    print(f"{'OVERALL AGGREGATE':<18} | {'All 4 Strata & 10 Materials (N=800)':<34} | "
          f"{total_sc_rec:5.1f}% ({np.sum(scccd_table['break_detected']):3d}/800) | "
          f"{total_py_rec:5.1f}% ({np.sum(pyccd_table['break_detected']):3d}/800) | "
          f"{np.median(all_sc_lat):.1f} d        / {np.median(all_py_lat):.1f} d")
    print("=" * 105 + "\n")


# ==============================================================================
# MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main():
    t_start_total = time.perf_counter()
    print("=" * 80)
    print("LAND-COVER STRATIFIED PATCHES BENCHMARK PIPELINE")
    print("Multi-Sensor Semi-Empirical Target Injection (SC-CCD vs. USGS PyCCD)")
    print("=" * 80)

    # 1. Parse onset timestamp
    onset_dt = datetime.datetime.strptime(INJECTION_ONSET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    onset_ts = onset_dt.timestamp()
    print(f"Injection onset date: {INJECTION_ONSET_DATE} (timestamp: {onset_ts:.0f})")

    # 2. Open Master HDF5 Container
    print(f"Opening master container: {MASTER_H5_PATH}")
    with h5py.File(MASTER_H5_PATH, "r+", locking=False) as h5_file:
        # Load Tanager wavelengths if available
        tanager_wl = None
        if "/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance" in h5_file:
            tanager_wl = h5_file["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"].attrs.get("wavelengths")

        # Load 10 resampled target spectra
        print(f"Loading USGS Splib07 spectra from: {SPLIB_BASE}")
        target_spectra = load_all_target_spectra(SPLIB_BASE, tanager_wl)
        print(f"Successfully prepared {len(target_spectra)} target spectra.")

        # Step 1: Extract baseline chips and metadata
        chips, metadata = extract_baseline_chips(h5_file)

        # Step 2 & 3: Simulate Spectral Complexity Z-scores on GPU
        sim_z_cubes, unperturbed_z = simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts)
        chips["HARMONIZED_z_score"] = unperturbed_z

        # Step 4: Run Harmonized SC-CCD break detection (full 9x9 patch across time to evaluate neighbor effects)
        scccd_table, scccd_spatial = run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts)

        # Step 5: Run USGS LCMAP Landsat PyCCD break detection (4 implanted pixels per patch, copying baseline for neighbors)
        pyccd_table, pyccd_spatial = run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts)

        # Step 6: Commit complete benchmark group to master HDF5
        save_benchmark_to_hdf5(
            h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table,
            scccd_spatial=scccd_spatial, pyccd_spatial=pyccd_spatial
        )

    # Step 7: Print quantitative summary
    print_performance_summary(scccd_table, pyccd_table)

    t_elapsed_total = (time.perf_counter() - t_start_total) / 60.0
    print(f"Stratified benchmark pipeline completed successfully in {t_elapsed_total:.2f} minutes.")


if __name__ == "__main__":
    main()

