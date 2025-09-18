#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

from ecg_pipeline import (
    PipelineConfig, load_config_yaml, ECGDenoisingPipeline, plot_signal_with_spans,
    load_ecg_npy, load_intervals, save_intervals_json, save_numpy_array
)

def build_argparser():
    ap = argparse.ArgumentParser(description="ECG Denoising Pipeline (memory-lean)")
    ap.add_argument("ecg", type=str, help=".npy file with ECG array (float or int)")
    ap.add_argument("gaps", type=str, help="Intervals file (json or txt) with gaps [start, end]")
    ap.add_argument("--config", type=str, default="", help="YAML config file (optional)")
    ap.add_argument("--fs", type=int, default=0, help="Override sampling rate in Hz")
    ap.add_argument("--output-dir", type=str, default="outputs", help="Directory to store results")
    ap.add_argument("--memory-lean", action="store_true", help="Enable memory-lean mode")
    ap.add_argument("--no-memory-lean", action="store_true", help="Disable memory-lean mode")
    ap.add_argument("--plot", action="store_true", help="Show plots interactively")
    ap.add_argument("--save-plots", action="store_true", help="Save plots as PNGs in output-dir")
    ap.add_argument("--write-intermediates", action="store_true", help="Save ecg_start and ecg_final arrays")
    return ap

def main():
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        cfg = load_config_yaml(args.config)
    else:
        cfg = PipelineConfig()

    if args.fs > 0:
        cfg.fs = args.fs
    if args.memory_lean:
        cfg.memory_lean = True
    if args.no_memory_lean:
        cfg.memory_lean = False

    ecg = load_ecg_npy(args.ecg)
    gaps = load_intervals(args.gaps, length=len(ecg))

    pipe = ECGDenoisingPipeline(cfg)
    res = pipe.run(ecg, gaps)

    # Save projections
    save_intervals_json(res.projected_to_orig["gaps"], out_dir / "gaps_on_orig.json")
    save_intervals_json(res.projected_to_orig["outliers"], out_dir / "outliers_on_orig.json")
    save_intervals_json(res.projected_to_orig["rdropouts"], out_dir / "rdropouts_on_orig.json")
    save_intervals_json(res.projected_to_orig["motions"], out_dir / "motions_on_orig.json")

    save_intervals_json(res.projected_to_start["outliers"], out_dir / "outliers_on_start.json")
    save_intervals_json(res.projected_to_start["rdropouts"], out_dir / "rdropouts_on_start.json")
    save_intervals_json(res.projected_to_start["motions"], out_dir / "motions_on_start.json")

    if args.write_intermediates:
        save_numpy_array(res.ecg_start, out_dir / "ecg_start.npy")
        save_numpy_array(res.ecg_final, out_dir / "ecg_final.npy")

    # Plots
    if args.save_plots or args.plot:
        plot_signal_with_spans(
            res.ecg_orig,
            {
                "gaps": res.projected_to_orig["gaps"],
                "outliers": res.projected_to_orig["outliers"],
                "rdropouts": res.projected_to_orig["rdropouts"],
                "motions": res.projected_to_orig["motions"],
            },
            cfg.fs,
            title="ECG Original with projected noisy segments",
            show=args.plot,
            save_path=str(out_dir / "plot_ecg_orig.png") if args.save_plots else None
        )

        plot_signal_with_spans(
            res.ecg_start,
            {
                "outliers": res.projected_to_start["outliers"],
                "rdropouts": res.projected_to_start["rdropouts"],
                "motions": res.projected_to_start["motions"],
            },
            cfg.fs,
            title="ECG Start (no gaps, filtered) with noisy segments",
            show=args.plot,
            save_path=str(out_dir / "plot_ecg_start.png") if args.save_plots else None
        )

    # Summary
    summary = {
        "len_original": int(len(res.ecg_orig)),
        "len_start": int(len(res.ecg_start)),
        "len_final": int(len(res.ecg_final)),
        "gaps_count": int(len(res.projected_to_orig["gaps"])),
        "outliers_count_start": int(len(res.projected_to_start["outliers"])),
        "rdropouts_count_start": int(len(res.projected_to_start["rdropouts"])),
        "motions_count_start": int(len(res.projected_to_start["motions"])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary))

if __name__ == "__main__":
    main()