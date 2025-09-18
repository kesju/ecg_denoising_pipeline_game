from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .config import PipelineConfig
from .index_map import Interval, IndexMap, normalize_intervals
from .steps import (
    remove_gaps, filter_ecg, detect_outliers, detect_rdropouts, detect_motions,
    remove_intervals_and_build_map
)

@dataclass
class PipelineResult:
    ecg_orig: np.ndarray
    ecg_start: np.ndarray
    ecg_final: np.ndarray
    # maps (forward)
    map_gaps: IndexMap             # orig -> no_gaps (== start before filtering length-wise)
    map_outliers: IndexMap          # start -> no_outliers
    map_rdropouts: IndexMap         # no_outliers -> no_rdropouts
    map_motions: IndexMap           # no_rdropouts -> final
    # detected intervals in their native stages
    gaps_indices: List[Interval]
    outliers_indices_start: List[Interval]
    rdropouts_indices_nout: List[Interval]
    motions_indices_nrd: List[Interval]
    # projections
    projected_to_orig: Dict[str, List[Interval]]
    projected_to_start: Dict[str, List[Interval]]

class ECGDenoisingPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def _compose_back_to_start(self) -> IndexMap:
        # Build map from start -> final
        m1 = self.map_outliers
        m2 = self.map_rdropouts
        m3 = self.map_motions
        return m1.compose(m2).compose(m3)

    def run(self, ecg_orig: np.ndarray, gaps_indices: List[Interval]) -> PipelineResult:
        cfg = self.cfg
        x_orig = np.asarray(ecg_orig).astype(float, copy=False)

        # 1) Remove gaps
        x_nogaps, map_gaps = remove_gaps(x_orig, normalize_intervals(gaps_indices, len(x_orig)))
        self.map_outliers = None  # type: ignore
        self.map_rdropouts = None  # type: ignore
        self.map_motions = None  # type: ignore

        # 2) Filter (length preserved)
        fcfg = cfg.filter
        if fcfg.enabled:
            x_start = filter_ecg(x_nogaps, cfg.fs, fcfg.method, fcfg.type, fcfg.lowcut, fcfg.highcut, fcfg.order)
        else:
            x_start = x_nogaps.copy()

        # Keep ecg_orig + ecg_start
        ecg_start = x_start.copy()

        # 3) Outliers: detect on ecg_start (no removal yet)
        oparams = cfg.outliers
        outliers_idx = []
        if oparams.enabled:
            outliers_idx = normalize_intervals(
                detect_outliers(x_start, oparams.z_thresh, oparams.min_len, oparams.merge_gap),
                len(x_start)
            )
        # Remove outliers to produce no_outliers
        x_no_outl, map_outliers = remove_intervals_and_build_map(x_start, outliers_idx)

        # Memory lean: free previous stage arrays no longer needed
        if cfg.memory_lean:
            del x_nogaps
            # keep x_start (ecg_start) in a separate var only

        # 4) R-dropouts: detect on x_no_outl
        rparams = cfg.rdropouts
        rdrop_idx = []
        if rparams.enabled:
            rdrop_idx = normalize_intervals(
                detect_rdropouts(x_no_outl, rparams.win, rparams.var_thresh, rparams.merge_gap),
                len(x_no_outl)
            )
        x_no_rdrop, map_rdrop = remove_intervals_and_build_map(x_no_outl, rdrop_idx)

        if cfg.memory_lean:
            del x_no_outl

        # 5) Motions: detect on x_no_rdrop
        mparams = cfg.motions
        motion_idx = []
        if mparams.enabled:
            motion_idx = normalize_intervals(
                detect_motions(x_no_rdrop, mparams.win, mparams.std_thresh, mparams.merge_gap),
                len(x_no_rdrop)
            )
        x_final, map_motions = remove_intervals_and_build_map(x_no_rdrop, motion_idx)

        if cfg.memory_lean:
            del x_no_rdrop

        # Compose projections
        # Project outliers (start) -> orig
        outl_on_orig = map_gaps.project_intervals_output_to_input(outliers_idx)
        # Project rdropouts (on no_outliers) -> start -> orig
        rdrop_on_start = map_outliers.project_intervals_output_to_input(rdrop_idx)
        rdrop_on_orig = map_gaps.project_intervals_output_to_input(rdrop_on_start)
        # Motions (on no_rdropouts) -> start -> orig
        mot_on_nout = map_rdrop.project_intervals_output_to_input(motion_idx)
        mot_on_start = map_outliers.project_intervals_output_to_input(mot_on_nout)
        mot_on_orig = map_gaps.project_intervals_output_to_input(mot_on_start)

        projected_to_orig = {
            "gaps": normalize_intervals(gaps_indices, len(x_orig)),
            "outliers": outl_on_orig,
            "rdropouts": rdrop_on_orig,
            "motions": mot_on_orig,
        }
        projected_to_start = {
            # gaps were removed -> no positions in start; keep empty by design
            "gaps": [],
            "outliers": outliers_idx,
            "rdropouts": rdrop_on_start,
            "motions": mot_on_start,
        }

        self.map_outliers = map_outliers
        self.map_rdropouts = map_rdrop
        self.map_motions = map_motions

        return PipelineResult(
            ecg_orig=x_orig.copy(),
            ecg_start=ecg_start.copy(),
            ecg_final=x_final.copy(),
            map_gaps=map_gaps,
            map_outliers=map_outliers,
            map_rdropouts=map_rdrop,
            map_motions=map_motions,
            gaps_indices=normalize_intervals(gaps_indices, len(x_orig)),
            outliers_indices_start=outliers_idx,
            rdropouts_indices_nout=rdrop_idx,
            motions_indices_nrd=motion_idx,
            projected_to_orig=projected_to_orig,
            projected_to_start=projected_to_start,
        )