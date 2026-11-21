"""
HarmonizedSC_run_pipeline.py
============================
Orchestrator for Spectral Complexity and Continuous Change Detection.

Serializes the 7-step processing chain:
    Step 1: HLS30_earthAccess_to_hdf5 (NASA HLS30 STAC Ingestion)
    Step 2: Tanager_STAC-imagery-downloader (Planet Tanager-1 STAC Ingestion)
    Step 3: HLST_constellation_to_hdf5 (Albers Orthorectification & Harmonization)
    Step 4: HLST_SC_calculations (Spectral Complexity)
    Step 5: pyccd_landsat_main (USGS LCMAP Landsat Temporal Regression)
    Step 6: harmonized_CCD_main (Harmonized Continuous Change Detection)
    Step 7: harmonized_CCD_vis (Continuous Change Detection Spatial Visualization)

CLI Usage Examples:
    # Run full end-to-end baseline pipeline (Steps 1-7):
    python HarmonizedSC_run_pipeline.py

    # Run specific step range:
    python HarmonizedSC_run_pipeline.py --steps 1-4
    python HarmonizedSC_run_pipeline.py --steps 5-7

    # Run change detection with optional injected data if present:
    python HarmonizedSC_run_pipeline.py --steps 5-7 --injected
"""

import os
import sys
import re
import argparse
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

# ==============================================================================
# Step Definitions & Pipeline Registry
# ==============================================================================
STEP_DEFINITIONS = {
    1: {
        "name": "HLS30_earthAccess_to_hdf5",
        "description": "NASA HLS30 STAC Download and HDF5 Ingestion (Native HLS L30 & S30)",
        "module": "HLS30_earthAccess_to_hdf5",
        "pipeline": "baseline",
    },
    2: {
        "name": "Tanager_STAC-imagery-downloader",
        "description": "Planet Tanager-1 STAC Basic Reflectance Download",
        "module": "Tanager_STAC-imagery-downloader",
        "pipeline": "baseline",
    },
    3: {
        "name": "HLST_constellation_to_hdf5",
        "description": "Albers Equal Area Constellation Orthorectification and Harmonization",
        "module": "HLST_constellation_to_hdf5",
        "pipeline": "baseline",
    },
    4: {
        "name": "HLST_SC_calculations",
        "description": "Spectral Complexity",
        "module": "HLST_SC_calculations",
        "pipeline": "baseline",
    },
    5: {
        "name": "pyccd_landsat_main",
        "description": "USGS LCMAP PyCCD Full-Spectrum Landsat Temporal Regression",
        "module": "pyccd_landsat_main",
        "pipeline": "baseline",
    },
    6: {
        "name": "harmonized_CCD_main",
        "description": "Harmonized Continuous Change Detection on SC Z-Score",
        "module": "harmonized_CCD_main",
        "pipeline": "baseline",
    },
    7: {
        "name": "harmonized_CCD_vis",
        "description": "Continuous Change Detection Spatial Visualization and Diagnostic Maps",
        "module": "harmonized_CCD_vis",
        "pipeline": "baseline",
    },
}

BASELINE_STEPS: List[int] = [1, 2, 3, 4, 5, 6, 7]
ALL_STEPS: List[int] = list(range(1, 8))


def parse_steps(
    steps_arg: Any = None,
    max_step: int = len(STEP_DEFINITIONS),
    injected_only: bool = False,
    baseline_only: bool = False,
    run_injected: bool = False,
) -> List[int]:
    """
    Parses step specification into a sorted list of unique integer step numbers.

    Parameters:
        steps_arg: String range ('1-7', '5-7'), comma-separated ('1,2,3'),
                   preset ('baseline', 'all'), list of ints or strings, or None.
        max_step: Maximum valid step integer (defaults to 7).
        injected_only: If True, executes change detection steps [5, 6, 7].
        baseline_only: If True, forces execution of baseline pipeline steps [1, 2, 3, 4, 5, 6, 7].
        run_injected: Passed through for compatibility.

    Returns:
        Sorted list of integer step numbers.
    """
    if baseline_only:
        return list(BASELINE_STEPS)
    if injected_only:
        return [5, 6, 7]

    if steps_arg is None:
        return list(range(1, max_step + 1))

    if not isinstance(steps_arg, (str, list, tuple)):
        raise TypeError(f"Steps argument must be string, list, or tuple, got {type(steps_arg).__name__}")

    if isinstance(steps_arg, str):
        cleaned = steps_arg.strip().lower()
        if cleaned in ("baseline", "all"):
            return list(range(1, max_step + 1))
        if cleaned == "injected":
            return [5, 6, 7]
        raw_tokens = [tok for tok in re.split(r"[\s,]+", steps_arg.strip()) if tok]
    else:
        raw_tokens = []
        for item in steps_arg:
            if isinstance(item, int):
                raw_tokens.append(str(item))
            elif isinstance(item, str):
                cleaned = item.strip().lower()
                if cleaned in ("baseline", "all"):
                    return list(range(1, max_step + 1))
                if cleaned == "injected":
                    return [5, 6, 7]
                parts = [tok for tok in re.split(r"[\s,]+", item.strip()) if tok]
                raw_tokens.extend(parts)
            else:
                raise TypeError(f"Invalid item in steps list: {type(item).__name__}")

    steps_set = set()
    for token in raw_tokens:
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Malformed range expression: '{token}'")
            try:
                start_val = int(bounds[0])
                end_val = int(bounds[1])
            except ValueError as e:
                raise ValueError(f"Non-integer value in range '{token}': {e}") from e

            if start_val > end_val:
                raise ValueError(f"Range start cannot exceed end: '{token}'")
            if start_val < 1 or end_val > max_step:
                raise ValueError(
                    f"Step out of bounds [1, {max_step}]: '{token}'. "
                    f"Note: Grid synthetic target injection method (create_injected_dataset) has been removed. "
                    f"Use generate_stratified_patch_benchmark.py for target injection benchmarking."
                )
            for s in range(start_val, end_val + 1):
                steps_set.add(s)
        else:
            try:
                s = int(token)
            except ValueError as e:
                raise ValueError(f"Invalid step identifier '{token}': {e}") from e
            if s < 1 or s > max_step:
                raise ValueError(
                    f"Step out of bounds [1, {max_step}]: {s}. "
                    f"Note: Grid synthetic target injection method (create_injected_dataset) has been removed. "
                    f"Use generate_stratified_patch_benchmark.py for target injection benchmarking."
                )
            steps_set.add(s)

    if not steps_set:
        raise ValueError("No valid steps resolved from input.")

    return sorted(steps_set)


def load_pipeline_config(config_path: Optional[str] = None, location: Optional[str] = None) -> Dict[str, Any]:
    """Loads location and pipeline configuration YAML file with fallback hierarchy."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                return loaded

    config_data = {
        "current_run": {"location": "Rochesterv2"},
        "locations": {
            "Rochesterv2": {
                "SOURCE_CACHE": None,
                "ROI_LON_MIN": -77.770166,
                "ROI_LON_MAX": -77.376776,
                "ROI_LAT_MIN": 42.961778,
                "ROI_LAT_MAX": 43.342135,
                "START_DATE": "2015-01-01",
                "END_DATE": "2026-06-01",
                "TANAGER_AVAILABLE": True,
            }
        }
    }
    return config_data


def run_pipeline(
    location: Optional[str] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2026-06-01",
    tile_size: int = 3,
    num_endmembers: int = 7,
    active_steps: Optional[Any] = None,
    steps: Optional[Any] = None,
    injected: bool = False,
    run_injected: bool = False,
    injected_only: bool = False,
    baseline_only: bool = False,
    h5_path: Optional[str] = None,
    dry_run: bool = False,
    skip_earthaccess: bool = False,
    skip_tanager: bool = False,
    config_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Executes the 7-step multi-sensor spectral complexity and change detection pipeline.

    Steps (1 - 7): STAC ingestion, orthorectification, SC, PyCCD, CCD, visualization.

    Parameters:
        location: Target location identifier (default 'Rochesterv2').
        start_date: Start date string (YYYY-MM-DD), default '2015-01-01'.
        end_date: End date string (YYYY-MM-DD), default '2026-06-01'.
        tile_size: Spatial sliding window size for spectral complexity calculation.
        num_endmembers: Number of endmembers for simplex extraction.
        active_steps: List or string of step numbers/ranges to execute (alias for 'steps').
        steps: Step range string or list (e.g. '1-7', '5-7', 'baseline').
        injected: If True, evaluates pre-existing injected datasets in change detection steps 5-7.
        run_injected: Alias for injected.
        injected_only: If True, runs change detection steps [5, 6, 7] with injected=True.
        baseline_only: If True, executes baseline pipeline steps [1, 2, 3, 4, 5, 6, 7].
        h5_path: Explicit path to master HDF5 container.
        dry_run: If True, validates configuration and step sequence without executing compute.
        skip_earthaccess: If True, skips Step 1 (NASA STAC download).
        skip_tanager: If True, skips Step 2 (Planet Tanager STAC download).
        config_path: Path to YAML pipeline configuration file.

    Returns:
        Dictionary containing execution summary and artifact paths.
    """
    if active_steps is None and steps is not None:
        active_steps = steps

    use_injected = injected or run_injected or injected_only

    resolved_steps = parse_steps(
        steps_arg=active_steps,
        max_step=len(STEP_DEFINITIONS),
        injected_only=injected_only,
        baseline_only=baseline_only,
        run_injected=use_injected,
    )

    if skip_earthaccess and 1 in resolved_steps:
        resolved_steps.remove(1)
    if skip_tanager and 2 in resolved_steps:
        resolved_steps.remove(2)

    config_data = load_pipeline_config(config_path=config_path, location=location)

    if location is None:
        location = config_data.get("current_run", {}).get("location", "Rochesterv2")

    if use_injected:
        pipeline_mode = "INJECTED CHANGE DETECTION MODE"
    else:
        pipeline_mode = "BASELINE PIPELINE (Steps 1-7)"

    print("=" * 70)
    print(" HARMONIZED SPECTRAL COMPLEXITY & CHANGE DETECTION PIPELINE")
    print("=" * 70)
    print(f" Target Location: {location}")
    print(f" Temporal Span:   {start_date} to {end_date}")
    print(f" Pipeline Scope:  {pipeline_mode}")
    print(f" Active Steps:    {resolved_steps}")
    print(f" Tile Size:       {tile_size}")
    print(f" Num Endmembers:  {num_endmembers}")
    print(f" Injected Mode:   {use_injected}")
    print(f" Dry Run Mode:    {dry_run}")
    print("=" * 70)

    pipeline_state: Dict[str, Any] = {
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "executed_steps": [],
        "artifacts": {},
        "status": "RUNNING",
        "dry_run": dry_run,
        "pipeline_mode": pipeline_mode,
    }

    # Resolve existing master container if starting downstream
    master_h5: Optional[str] = None
    candidate_paths = [
        f"C:/satelliteImagery/HLST30/HLST_{location}.h5",
    ]
    if h5_path is not None:
        candidate_paths.insert(0, h5_path)

    for cand in candidate_paths:
        if cand and os.path.exists(cand):
            master_h5 = cand
            break

    if master_h5:
        print(f" Resolved Master HDF5: {master_h5}")

    for step_num in resolved_steps:
        step_info = STEP_DEFINITIONS[step_num]
        step_name = step_info["name"]
        step_desc = step_info["description"]
        step_pipe = step_info["pipeline"].upper()

        print(f"\n{'-' * 60}")
        print(f"Step {step_num} [{step_pipe}]: {step_name}")
        print(f"Purpose: {step_desc}")
        print(f"{'-' * 60}")

        if dry_run:
            print(f"  [DRY-RUN] Step {step_num} ({step_name}) validated.")
            pipeline_state["executed_steps"].append(step_num)
            continue

        # Step 1: NASA HLS30 STAC Ingestion
        if step_num == 1:
            hls_mod = importlib.import_module("HLS30_earthAccess_to_hdf5")
            native_h5 = hls_mod.main(
                target_location=location,
                start_date=start_date,
                end_date=end_date,
                config_override=config_data,
            )
            pipeline_state["artifacts"]["native_h5"] = native_h5
            print(f"Step 1 Complete: {native_h5}")

        # Step 2: Tanager STAC Ingestion
        elif step_num == 2:
            tanager_mod = importlib.import_module("Tanager_STAC-imagery-downloader")
            total_scenes = tanager_mod.main(
                start_date=start_date,
                end_date=end_date,
            )
            pipeline_state["artifacts"]["tanager_scenes"] = total_scenes
            print(f"Step 2 Complete: Processed {total_scenes} Tanager scenes.")

        # Step 3: Albers Orthorectification and Harmonization
        elif step_num == 3:
            harm_mod = importlib.import_module("HLST_constellation_to_hdf5")
            input_h5 = pipeline_state["artifacts"].get("native_h5")
            master_h5 = harm_mod.main(
                target_location=location,
                input_native_h5=input_h5,
                config_override=config_data,
            )
            pipeline_state["artifacts"]["master_h5"] = master_h5
            print(f"Step 3 Complete: {master_h5}")

        # Step 4: Baseline Spectral Complexity
        elif step_num == 4:
            if master_h5 is None:
                raise RuntimeError(
                    f"No master HDF5 container available for Step 4. "
                    f"Run Step 3 first or specify an existing container with --h5-path (checked: {candidate_paths})."
                )
            sc_mod = importlib.import_module("HLST_SC_calculations")
            sc_mod.process_ard_cube(
                master_h5,
                tile_size=tile_size,
                num_endmembers=num_endmembers,
            )
            pipeline_state["artifacts"]["sc_h5"] = master_h5
            print(f"Step 4 Complete: Updated baseline SC metric in-place in {master_h5}")

        # Step 5: USGS LCMAP PyCCD Full-Spectrum Landsat
        elif step_num == 5:
            if master_h5 is None:
                raise RuntimeError(
                    f"No master HDF5 container available for Step 5. "
                    f"Run Step 3 first or specify an existing container with --h5-path (checked: {candidate_paths})."
                )
            pyccd_mod = importlib.import_module("pyccd_landsat_main")
            print(f"  Running PyCCD on Landsat surface reflectance (injected={use_injected})...")
            pyccd_res_h5 = pyccd_mod.main(
                location=location,
                h5_path=master_h5,
                injected=use_injected,
            )
            pipeline_state["artifacts"]["pyccd_h5"] = pyccd_res_h5
            print(f"Step 5 Complete: PyCCD Landsat run complete ({pyccd_res_h5}).")

        # Step 6: Harmonized Continuous Change Detection (SC Z-Score)
        elif step_num == 6:
            if master_h5 is None:
                raise RuntimeError(
                    f"No master HDF5 container available for Step 6. "
                    f"Run Step 3 first or specify an existing container with --h5-path (checked: {candidate_paths})."
                )
            ccd_mod = importlib.import_module("harmonized_CCD_main")

            print(f"  Running Harmonized CCD on Spectral Complexity Z-Score (injected={use_injected})...")
            ccd_mod.main(
                location=location,
                h5_path=master_h5,
                target_metric="sliding_volume_z_score",
                injected=use_injected,
                launch_vis=False,
            )

        # Step 7: Continuous Change Detection Visualization
        elif step_num == 7:
            if master_h5 is None:
                raise RuntimeError(
                    f"No master HDF5 container available for Step 7. "
                    f"Run Step 3 first or specify an existing container with --h5-path (checked: {candidate_paths})."
                )
            vis_mod = importlib.import_module("harmonized_CCD_vis")
            pyccd_vis_mod = importlib.import_module("pyccd_landsat_vis")
            diag_pngs = []

            # Harmonized CCD overlays (SC)
            for metric in ["SC"]:
                out_png = vis_mod.get_default_png_name(location=location, target_metric=metric, injected=use_injected)
                try:
                    vis_mod.plot_spatial_anomaly_overlay(
                        source_h5_path=master_h5,
                        target_metric=metric,
                        injected=use_injected,
                        output_png=out_png,
                        interactive=False,
                    )
                    diag_pngs.append(out_png)
                    print(f"  Exported Harmonized CCD overlay: {out_png}")
                except Exception as e:
                    print(f"  Warning: Could not render visualization for {metric}: {e}")

            # PyCCD Landsat overlay
            out_png = pyccd_vis_mod.get_default_png_name(location=location, injected=use_injected)
            try:
                pyccd_vis_mod.plot_spatial_change_map(
                    source_h5_path=master_h5,
                    group_name="INJECTED_PYCCD_Results" if use_injected else "PYCCD_Results",
                    injected=use_injected,
                    output_png=out_png,
                    interactive=False,
                )
                diag_pngs.append(out_png)
                print(f"  Exported PyCCD overlay: {out_png}")
            except Exception as e:
                print(f"  Warning: Could not render PyCCD visualization: {e}")

            pipeline_state["artifacts"].setdefault("diagnostic_pngs", []).extend(diag_pngs)
            print("Step 7 Complete: Diagnostic maps exported.")

        pipeline_state["executed_steps"].append(step_num)

    pipeline_state["status"] = "SUCCESS"
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(f" Completed Steps: {pipeline_state['executed_steps']}")
    print("=" * 70)
    return pipeline_state


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command line argument parser for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Harmonized Spectral Complexity & Continuous Change Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step Reference:
  1: HLS30_earthAccess_to_hdf5       NASA HLS30 STAC Ingestion
  2: Tanager_STAC-imagery-downloader Planet Tanager-1 STAC Ingestion
  3: HLST_constellation_to_hdf5      Albers Orthorectification & Harmonization
  4: HLST_SC_calculations            Spectral Complexity
  5: pyccd_landsat_main              USGS LCMAP Landsat Temporal Regression
  6: harmonized_CCD_main             Harmonized Continuous Change Detection
  7: harmonized_CCD_vis              Continuous Change Detection Visualizations
        """,
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Target location identifier (default 'Rochesterv2')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Temporal acquisition start date (YYYY-MM-DD, default 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-06-01",
        help="Temporal acquisition end date (YYYY-MM-DD, default 2026-06-01)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=3,
        help="Sliding window tile size for spectral complexity calculation (default 3)",
    )
    parser.add_argument(
        "--num-endmembers",
        type=int,
        default=7,
        help="Number of endmembers to extract in spectral complexity (default 7)",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default=None,
        help="Explicit path to master HDF5 container (e.g. C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="Step numbers, ranges, or preset ('1-7', '5-7', 'baseline', 'all')",
    )
    parser.add_argument(
        "--injected-only",
        action="store_true",
        default=False,
        help="Run change detection steps [5, 6, 7] with injected=True",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        default=False,
        help="Run baseline pipeline steps (Steps 1-7)",
    )
    parser.add_argument(
        "--injected",
        action="store_true",
        default=False,
        help="Run change detection steps on pre-existing injected datasets (default False)",
    )
    parser.add_argument(
        "--no-injected",
        action="store_false",
        dest="injected",
        help="Run change detection on baseline datasets (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate step sequence and configuration without executing compute",
    )
    parser.add_argument(
        "--skip-earthaccess",
        action="store_true",
        default=False,
        help="Skip NASA HLS30 EarthAccess STAC download (Step 1)",
    )
    parser.add_argument(
        "--skip-tanager",
        action="store_true",
        default=False,
        help="Skip Planet Tanager-1 STAC download (Step 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sliding_volume_z_score",
        help="Target metric for change detection (default 'sliding_volume_z_score')",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point returning integer status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            location=args.location,
            start_date=args.start_date,
            end_date=args.end_date,
            tile_size=args.tile_size,
            num_endmembers=args.num_endmembers,
            h5_path=args.h5_path,
            active_steps=args.steps,
            injected=args.injected,
            injected_only=args.injected_only,
            baseline_only=args.baseline_only,
            dry_run=args.dry_run,
            skip_earthaccess=args.skip_earthaccess,
            skip_tanager=args.skip_tanager,
            config_path=args.config,
        )
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Pipeline execution terminated with failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
