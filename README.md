# ECG Denoising Pipeline (Memory‑Lean)

A modular NumPy/Matplotlib project that removes gaps, filters the signal, and imitates detection/removal of outliers, R‑dropouts, and motion artifacts. It builds index mappings to project detected fragments back to either the original signal or the filtered starting signal.

## Structure

```
ecg_denoising_project/
├── ecg_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── index_map.py
│   ├── steps.py
│   ├── pipeline.py
│   ├── plotting.py
│   └── io_utils.py
├── cli.py
├── cli.ipynb
├── example_data/
│   ├── ecg_orig.npy
│   ├── gaps_indices.json
│   └── pipeline_config.yaml
├── pyproject.toml
└── README.md
```

## Install (editable)

```bash
pip install -e .
```

## CLI usage

```bash
# Using bundled example data
python cli.py example_data/ecg_orig.npy example_data/gaps_indices.json --config example_data/pipeline_config.yaml --save-plots --write-intermediates --output-dir outputs
```

This will save:
- `outputs/summary.json`
- projections to original: `gaps_on_orig.json`, `outliers_on_orig.json`, `rdropouts_on_orig.json`, `motions_on_orig.json`
- projections to start: `outliers_on_start.json`, `rdropouts_on_start.json`, `motions_on_start.json`
- optional plots: `plot_ecg_orig.png`, `plot_ecg_start.png`
- optional arrays: `ecg_start.npy`, `ecg_final.npy`

## Notes

- Filtering uses **NeuroKit2** if you set `method: neurokit` in the config and the library is installed; otherwise it falls back to a SciPy Butterworth implementation.
- The detections are **mock** (simple heuristics) meant to be replaced with your true algorithms.
- Memory‑lean mode releases intermediate arrays once the next stage is produced while keeping `ecg_orig`, `ecg_start`, and `ecg_final` for plotting.