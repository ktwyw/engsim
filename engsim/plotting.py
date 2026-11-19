"""Shared publication Matplotlib style and panel helpers for engsim."""

from __future__ import annotations


def use_style() -> None:
    """Apply the standard publication Matplotlib style used across the book."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
            "errorbar.capsize": 2.5,
            "mathtext.fontset": "cm",
            "figure.constrained_layout.use": True,
        }
    )


def panel_label(ax, text: str, x: float = 0.03, y: float = 0.95, **kwargs) -> None:
    """Place a bold panel label such as '(a)' in the upper-left corner of an Axes."""
    opts = {
        "transform": getattr(ax, "transAxes", None),
        "fontsize": 11,
        "fontweight": "bold",
        "va": "top",
        "ha": "left",
    }
    opts.update(kwargs)
    ax.text(x, y, text, **opts)
