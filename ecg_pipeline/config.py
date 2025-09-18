from dataclasses import dataclass
from typing import Optional, Literal
import yaml

@dataclass
class FilterParams:
    enabled: bool = True
    method: Literal["butter", "neurokit"] = "butter"  # 'neurokit' tries neurokit2 if available
    type: Literal["highpass", "bandpass"] = "bandpass"
    lowcut: Optional[float] = 0.5     # Hz (used in highpass or bandpass)
    highcut: Optional[float] = 40.0   # Hz (used only in bandpass)
    order: int = 4

@dataclass
class OutliersParams:
    enabled: bool = True
    z_thresh: float = 3.5
    min_len: int = 8            # samples to expand each detection
    merge_gap: int = 10         # merge close intervals closer than this

@dataclass
class RDropoutsParams:
    enabled: bool = True
    win: int = 40               # samples
    var_thresh: float = 1e-5    # low variance implies potential dropout
    merge_gap: int = 10

@dataclass
class MotionsParams:
    enabled: bool = True
    win: int = 20
    std_thresh: float = 0.15    # high std implies motion
    merge_gap: int = 10

@dataclass
class PipelineConfig:
    fs: int = 200
    memory_lean: bool = True
    filter: FilterParams = FilterParams()
    outliers: OutliersParams = OutliersParams()
    rdropouts: RDropoutsParams = RDropoutsParams()
    motions: MotionsParams = MotionsParams()

def load_config_yaml(path: str) -> PipelineConfig:
    """Load config from a YAML file into PipelineConfig dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def merge_dataclass(dc_cls, values):
        obj = dc_cls()
        for k, v in (values or {}).items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    filter_cfg = merge_dataclass(FilterParams, data.get("filter"))
    outliers_cfg = merge_dataclass(OutliersParams, data.get("outliers"))
    rdrop_cfg = merge_dataclass(RDropoutsParams, data.get("rdropouts"))
    motions_cfg = merge_dataclass(MotionsParams, data.get("motions"))

    cfg = PipelineConfig(
        fs=data.get("fs", 200),
        memory_lean=data.get("memory_lean", True),
        filter=filter_cfg,
        outliers=outliers_cfg,
        rdropouts=rdrop_cfg,
        motions=motions_cfg,
    )
    return cfg