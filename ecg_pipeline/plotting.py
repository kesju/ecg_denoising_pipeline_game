from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .index_map import Interval

def _span(ax, spans: List[Interval], label: str, alpha: float = 0.3):
    if not spans:
        return
    for (s, e) in spans:
        ax.axvspan(s, e, alpha=alpha, label=label)

def plot_signal_with_spans(
    x: np.ndarray,
    spans_dict: Dict[str, List[Interval]],
    fs: int,
    title: str = "",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Plot signal with colored spans for different labels.
    Note: colors are left to Matplotlib defaults; labels will auto-legend.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, linewidth=1.0)
    for label, spans in spans_dict.items():
        _span(ax, spans, label=label, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Samples (fs=%d Hz)" % fs)
    ax.set_ylabel("Amplitude")
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate labels
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=4, fontsize=8, framealpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)