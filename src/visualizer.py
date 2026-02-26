from __future__ import annotations
import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    _PLOTLY_AVAILABLE = False

try:
    from analyzer import LayerStats, compute_layer_stats
    _ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from src.analyzer import LayerStats, compute_layer_stats
        _ANALYZER_AVAILABLE = True
    except ImportError:
        LayerStats = None
        compute_layer_stats = None
        _ANALYZER_AVAILABLE = False

_PALETTE_DIVERGING = "RdBu_r"   # для delta / effect_size (negative=blue, positive=red)
_PALETTE_SEQUENTIAL = "YlOrRd"  # для abs-метрик (eta2, snr, abs_delta)
_PALETTE_POS = "Reds"
_PALETTE_NEG = "Blues"

_FIG_DPI = 150
_STYLE = "dark_background"

plt.rcParams.update({
    "font.family": "monospace",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _guess_repo_root() -> str:
    """src/visualizer.py -> repo_root = parent of src."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_npz(run_dir: str, layer: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Спочатку шукаємо .npz з analyzer (outputs/reports/analyzer_<run_name>_layer_<L>.npz),
    якщо немає — завантажуємо сирий .npy і рахуємо статистику на льоту.
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    repo_root = _guess_repo_root()

    npz_path = os.path.join(
        repo_root, "outputs", "reports", f"analyzer_{run_name}_layer_{layer}.npz"
    )
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}

    # Fallback: рахуємо самі (потребує analyzer)
    if not _ANALYZER_AVAILABLE:
        return None

    acts_path = os.path.join(run_dir, f"mlp_layer_{layer}.npy")
    labels_path = os.path.join(run_dir, "labels.npy")
    if not (os.path.isfile(acts_path) and os.path.isfile(labels_path)):
        return None

    acts = np.load(acts_path)
    labels = np.load(labels_path).astype(np.int64).reshape(-1)
    st = compute_layer_stats(acts, labels, layer=layer)

    return {
        "mean_pos": st.mean_pos,
        "mean_neg": st.mean_neg,
        "delta": st.delta,
        "abs_delta": st.abs_delta,
        "std_all": st.std_all,
        "effect_size": st.effect_size,
        "eta2": st.eta2,
        "snr": st.snr,
    }


def _find_layers(run_dir: str) -> List[int]:
    layers = []
    for f in os.listdir(run_dir):
        if f.startswith("mlp_layer_") and f.endswith(".npy"):
            try:
                layers.append(int(f[len("mlp_layer_"):-len(".npy")]))
            except ValueError:
                pass
    return sorted(layers)


def _build_matrix(
    layers: List[int],
    run_dir: str,
    metric: str,
    max_neurons: int = 512,
) -> Optional[np.ndarray]:
    """
    Будує матрицю (n_layers, max_neurons) для заданої метрики.
    Якщо нейронів більше max_neurons — субсемплює рівномірно.
    """
    rows = []
    for layer in layers:
        data = _load_npz(run_dir, layer)
        if data is None or metric not in data:
            return None
        arr = data[metric].astype(np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()
        if arr.shape[0] > max_neurons:
            idx = np.linspace(0, arr.shape[0] - 1, max_neurons, dtype=int)
            arr = arr[idx]
        rows.append(arr)

    if not rows:
        return None

    min_len = min(r.shape[0] for r in rows)
    matrix = np.stack([r[:min_len] for r in rows], axis=0)  # (n_layers, neurons)
    return matrix


def _layer_profile(layers: List[int], run_dir: str) -> Dict[str, np.ndarray]:
    """Агрегує метрики по шарах (mean / max)."""
    profiles: Dict[str, List[float]] = {
        "mean_abs_delta": [],
        "max_abs_delta": [],
        "mean_effect_size": [],
        "max_effect_size": [],
        "mean_eta2": [],
        "mean_snr": [],
    }

    for layer in layers:
        data = _load_npz(run_dir, layer)
        if data is None:
            for k in profiles:
                profiles[k].append(0.0)
            continue

        profiles["mean_abs_delta"].append(float(data["abs_delta"].mean()))
        profiles["max_abs_delta"].append(float(data["abs_delta"].max()))
        profiles["mean_effect_size"].append(float(np.abs(data["effect_size"]).mean()))
        profiles["max_effect_size"].append(float(np.abs(data["effect_size"]).max()))
        profiles["mean_eta2"].append(float(data["eta2"].mean()))
        profiles["mean_snr"].append(float(data["snr"].mean()))

    return {k: np.array(v) for k, v in profiles.items()}


def _get_top_neurons(
    layers: List[int],
    run_dir: str,
    metric: str = "effect_size",
    top_n: int = 30,
) -> List[Dict]:
    """Глобальний топ-N нейронів за абсолютним значенням метрики."""
    records = []
    for layer in layers:
        data = _load_npz(run_dir, layer)
        if data is None or metric not in data:
            continue
        arr = np.abs(data[metric])
        for i, v in enumerate(arr):
            records.append({"layer": layer, "neuron": i, "score": float(v)})

    records.sort(key=lambda r: r["score"], reverse=True)
    return records[:top_n]

def plot_heatmap(
    matrix: np.ndarray,
    layers: List[int],
    title: str,
    out_path: str,
    cmap: str = _PALETTE_DIVERGING,
    center: Optional[float] = 0.0,
    xlabel: str = "Neuron index (subsampled)",
    ylabel: str = "Layer",
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Базова теплова карта (layers × neurons)."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)

        norm = None
        if center is not None and cmap == _PALETTE_DIVERGING:
            vmax = float(np.abs(matrix).max()) or 1.0
            vmin = -vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            norm=norm,
            yticklabels=[f"L{l}" for l in layers],
            xticklabels=False,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, pad=10, fontsize=13, color="white")
        ax.set_xlabel(xlabel, color="#aaaaaa")
        ax.set_ylabel(ylabel, color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")

        plt.tight_layout()
        plt.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"  [viz] saved: {os.path.basename(out_path)}")


def plot_pos_neg_comparison(
    layers: List[int],
    run_dir: str,
    out_path: str,
    max_neurons: int = 256,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """Порівняльна карта mean_pos vs mean_neg (side-by-side)."""
    mat_pos = _build_matrix(layers, run_dir, "mean_pos", max_neurons=max_neurons)
    mat_neg = _build_matrix(layers, run_dir, "mean_neg", max_neurons=max_neurons)
    if mat_pos is None or mat_neg is None:
        print("  [viz] WARNING: pos/neg comparison: data unavailable")
        return

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=_FIG_DPI)

        for ax, mat, title, cmap in zip(
            axes,
            [mat_pos, mat_neg],
            ["Позитивні промпти (mean_pos)", "Негативні промпти (mean_neg)"],
            [_PALETTE_POS, _PALETTE_NEG],
        ):
            vmax = float(np.abs(mat).max()) or 1.0
            sns.heatmap(
                mat,
                ax=ax,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                yticklabels=[f"L{l}" for l in layers],
                xticklabels=False,
                linewidths=0,
                rasterized=True,
            )
            ax.set_title(title, fontsize=11, color="white")
            ax.set_xlabel("Нейрон (субсемпл)", color="#aaaaaa")
            ax.set_ylabel("Шар", color="#aaaaaa")
            ax.tick_params(colors="#aaaaaa")

        fig.suptitle(
            "Порівняння середніх активацій: Positive vs Negative",
            fontsize=13, color="white", y=1.01,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"  [viz] saved: {os.path.basename(out_path)}")


def plot_top_neurons_bar(
    layers: List[int],
    run_dir: str,
    out_path: str,
    top_n: int = 30,
    metric: str = "effect_size",
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Горизонтальний барплот топ-N нейронів з колірним кодуванням шару."""
    records = _get_top_neurons(layers, run_dir, metric=metric, top_n=top_n)
    if not records:
        print("  [viz] WARNING: top neurons bar: no data")
        return

    labels_bar = [f"L{r['layer']}·N{r['neuron']}" for r in records]
    scores = [r["score"] for r in records]
    layer_ids = [r["layer"] for r in records]

    cmap_layers = matplotlib.colormaps.get_cmap("plasma").resampled(len(layers) + 1)
    layer_to_color = {l: cmap_layers(i) for i, l in enumerate(layers)}
    colors = [layer_to_color[lid] for lid in layer_ids]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)
        ax.barh(range(len(records)), scores[::-1], color=colors[::-1])
        ax.set_yticks(range(len(records)))
        ax.set_yticklabels(labels_bar[::-1], fontsize=7)
        ax.set_xlabel(f"|{metric}|", color="#aaaaaa")
        ax.set_title(
            f"Топ-{top_n} нейронів за |{metric}| (глобально по всіх шарах)",
            color="white", fontsize=12,
        )
        ax.tick_params(colors="#aaaaaa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Легенда шарів
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=layer_to_color[l], label=f"Layer {l}")
            for l in layers
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
                  facecolor="#222", labelcolor="white")

        plt.tight_layout()
        plt.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"  [viz] saved: {os.path.basename(out_path)}")


def plot_layer_profile(
    layers: List[int],
    run_dir: str,
    out_path: str,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Лінійні графіки агрегованих метрик по шарах."""
    profile = _layer_profile(layers, run_dir)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=_FIG_DPI)

        configs = [
            (axes[0], ["mean_abs_delta", "max_abs_delta"], "Abs Delta", ["#FF6B6B", "#FF9999"]),
            (axes[1], ["mean_effect_size", "max_effect_size"], "Cohen's d", ["#4ECDC4", "#A8EDEA"]),
            (axes[2], ["mean_eta2", "mean_snr"], "η² та SNR", ["#FFE66D", "#A8D8EA"]),
        ]

        x = np.array(layers)
        for ax, metrics, title, colors_line in configs:
            for metric, color in zip(metrics, colors_line):
                vals = profile[metric]
                ax.plot(x, vals, color=color, lw=2, marker="o", ms=4, label=metric)
                ax.fill_between(x, 0, vals, alpha=0.15, color=color)
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xlabel("Шар", color="#aaaaaa")
            ax.legend(fontsize=6, facecolor="#222", labelcolor="white")
            ax.tick_params(colors="#aaaaaa")
            ax.set_xticks(layers)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Профіль метрик чутливості по шарах GPT-2", fontsize=13, color="white")
        plt.tight_layout()
        plt.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"  [viz] saved: {os.path.basename(out_path)}")


def plot_summary_dashboard(
    layers: List[int],
    run_dir: str,
    out_path: str,
    top_n: int = 20,
    max_neurons: int = 256,
) -> None:
    """Зведений дашборд — 6 панелей на одному полотні."""
    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(20, 14), dpi=_FIG_DPI)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        run_name = os.path.basename(os.path.normpath(run_dir))
        fig.suptitle(
            f"LLM Emotion Interpretability — Звіт\n{run_name}",
            fontsize=15, color="white", y=0.98,
        )

        # панель 1: delta heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        mat_delta = _build_matrix(layers, run_dir, "delta", max_neurons=max_neurons)
        if mat_delta is not None:
            vmax = float(np.abs(mat_delta).max()) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            sns.heatmap(mat_delta, ax=ax1, cmap=_PALETTE_DIVERGING, norm=norm,
                        yticklabels=[f"L{l}" for l in layers], xticklabels=False,
                        linewidths=0, rasterized=True)
            ax1.set_title("Δ = mean_pos − mean_neg (по нейронах)", color="white")
            ax1.tick_params(colors="#aaaaaa")

        # панель 2: effect_size heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        mat_d = _build_matrix(layers, run_dir, "effect_size", max_neurons=max_neurons)
        if mat_d is not None:
            vmax = float(np.abs(mat_d).max()) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            sns.heatmap(mat_d, ax=ax2, cmap=_PALETTE_DIVERGING, norm=norm,
                        yticklabels=[f"L{l}" for l in layers], xticklabels=False,
                        linewidths=0, rasterized=True)
            ax2.set_title("Cohen's d", color="white")
            ax2.tick_params(colors="#aaaaaa")

        # панель 3: mean_pos
        ax3 = fig.add_subplot(gs[1, 0])
        mat_pos = _build_matrix(layers, run_dir, "mean_pos", max_neurons=max_neurons)
        if mat_pos is not None:
            sns.heatmap(mat_pos, ax=ax3, cmap=_PALETTE_POS,
                        yticklabels=[f"L{l}" for l in layers], xticklabels=False,
                        linewidths=0, rasterized=True)
            ax3.set_title("mean_pos (Positive)", color="white")
            ax3.tick_params(colors="#aaaaaa")

        # панель 4: mean_neg
        ax4 = fig.add_subplot(gs[1, 1])
        mat_neg = _build_matrix(layers, run_dir, "mean_neg", max_neurons=max_neurons)
        if mat_neg is not None:
            sns.heatmap(mat_neg, ax=ax4, cmap=_PALETTE_NEG,
                        yticklabels=[f"L{l}" for l in layers], xticklabels=False,
                        linewidths=0, rasterized=True)
            ax4.set_title("mean_neg (Negative)", color="white")
            ax4.tick_params(colors="#aaaaaa")

        # панель 5: layer profile
        ax5 = fig.add_subplot(gs[1, 2])
        profile = _layer_profile(layers, run_dir)
        x = np.array(layers)
        ax5.plot(x, profile["mean_effect_size"], color="#4ECDC4", lw=2, marker="o", ms=4, label="mean |d|")
        ax5.plot(x, profile["max_effect_size"], color="#A8EDEA", lw=1.5, marker="s", ms=3, label="max |d|")
        ax5.fill_between(x, 0, profile["mean_effect_size"], alpha=0.15, color="#4ECDC4")
        ax5.set_title("Cohen's d по шарах", color="white")
        ax5.set_xticks(layers)
        ax5.tick_params(colors="#aaaaaa")
        ax5.legend(fontsize=7, facecolor="#222", labelcolor="white")
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)

        # панель 6: top neurons bar
        ax6 = fig.add_subplot(gs[2, :])
        records = _get_top_neurons(layers, run_dir, metric="effect_size", top_n=top_n)
        if records:
            bar_labels = [f"L{r['layer']}·N{r['neuron']}" for r in records]
            bar_scores = [r["score"] for r in records]
            bar_layer_ids = [r["layer"] for r in records]
            cmap_l = matplotlib.colormaps.get_cmap("plasma").resampled(len(layers) + 1)
            l2c = {l: cmap_l(i) for i, l in enumerate(layers)}
            bar_colors = [l2c[lid] for lid in bar_layer_ids]
            ax6.bar(range(len(records)), bar_scores, color=bar_colors)
            ax6.set_xticks(range(len(records)))
            ax6.set_xticklabels(bar_labels, rotation=70, fontsize=6)
            ax6.set_title(f"Топ-{top_n} нейронів за |effect_size| (глобально)", color="white")
            ax6.set_ylabel("|Cohen's d|", color="#aaaaaa")
            ax6.tick_params(colors="#aaaaaa")
            ax6.spines["top"].set_visible(False)
            ax6.spines["right"].set_visible(False)

        plt.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  [viz] saved: {os.path.basename(out_path)}")

# хтмл інтерактивні звіти
def plot_interactive_heatmap(
    matrix: np.ndarray,
    layers: List[int],
    title: str,
    out_path: str,
    colorscale: str = "RdBu",
    zmid: float = 0.0,
) -> None:
    """Зберігає інтерактивну Plotly теплову карту у HTML."""
    if not _PLOTLY_AVAILABLE:
        print("  [viz] WARNING: plotly not installed, HTML output disabled")
        return

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        zmid=zmid,
        y=[f"Layer {l}" for l in layers],
        colorbar=dict(title="value"),
        hovertemplate="Layer: %{y}<br>Neuron: %{x}<br>Value: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Neuron index (subsampled)",
        yaxis_title="Layer",
        template="plotly_dark",
        height=500,
    )
    fig.write_html(out_path)
    print(f"  [viz] saved: {os.path.basename(out_path)}")

def plot_interactive_top_neurons(
    layers: List[int],
    run_dir: str,
    out_path: str,
    top_n: int = 50,
    metric: str = "effect_size",
) -> None:
    """Інтерактивний барплот топ-N нейронів у Plotly (HTML)."""
    if not _PLOTLY_AVAILABLE:
        return

    records = _get_top_neurons(layers, run_dir, metric=metric, top_n=top_n)
    if not records:
        return

    bar_labels = [f"L{r['layer']}·N{r['neuron']}" for r in records]
    bar_scores = [r["score"] for r in records]
    bar_layers = [r["layer"] for r in records]

    fig = px.bar(
        x=bar_labels,
        y=bar_scores,
        color=bar_layers,
        labels={"x": "Neuron (Layer·Index)", "y": f"|{metric}|", "color": "Layer"},
        title=f"Топ-{top_n} нейронів за |{metric}| — інтерактивний графік",
        template="plotly_dark",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(height=500)
    fig.write_html(out_path)
    print(f"  [viz] saved: {os.path.basename(out_path)}")

class Visualizer:

    def __init__(
        self,
        run_dir: str,
        out_subdir: Optional[str] = None,
        top_n: int = 30,
        max_neurons: int = 512,
        generate_html: bool = True,
    ) -> None:
        self.run_dir = os.path.abspath(run_dir)
        self.run_name = os.path.basename(os.path.normpath(run_dir))
        self.top_n = top_n
        self.max_neurons = max_neurons
        self.generate_html = generate_html and _PLOTLY_AVAILABLE

        repo_root = _guess_repo_root()
        if out_subdir is None:
            out_subdir = os.path.join("outputs", "heatmaps", self.run_name)
        self.out_dir = os.path.join(repo_root, out_subdir)
        _safe_makedirs(self.out_dir)

        self.layers = _find_layers(self.run_dir)
        if not self.layers:
            raise FileNotFoundError(f"No mlp_layer_*.npy files found in: {run_dir}")

        print(f"\n[viz] run_dir   : {self.run_dir}")
        print(f"[viz] run_name  : {self.run_name}")
        print(f"[viz] layers    : {self.layers}")
        print(f"[viz] out_dir   : {self.out_dir}")
        print(f"[viz] plotly    : {'yes' if _PLOTLY_AVAILABLE else 'no (not installed, HTML disabled)'}")

    def _p(self, filename: str) -> str:
        return os.path.join(self.out_dir, filename)

    def heatmap_delta(self) -> None:
        """01 — теплова карта Delta."""
        mat = _build_matrix(self.layers, self.run_dir, "delta", self.max_neurons)
        if mat is None:
            print("  [viz] WARNING: delta: no data"); return
        plot_heatmap(mat, self.layers, "Δ = mean_pos − mean_neg (по нейронах GPT-2)",
                     self._p("01_delta_heatmap.png"), cmap=_PALETTE_DIVERGING, center=0.0)

    def heatmap_effect_size(self) -> None:
        """02 — теплова карта Cohen's d."""
        mat = _build_matrix(self.layers, self.run_dir, "effect_size", self.max_neurons)
        if mat is None:
            print("  [viz] WARNING: effect_size: no data"); return
        plot_heatmap(mat, self.layers, "Cohen's d (effect_size) — чутливість нейронів до емоції",
                     self._p("02_effect_size_heatmap.png"), cmap=_PALETTE_DIVERGING, center=0.0)

    def bar_top_neurons(self) -> None:
        """03 — барплот топ-N нейронів."""
        plot_top_neurons_bar(self.layers, self.run_dir, self._p("03_top_neurons_bar.png"),
                             top_n=self.top_n)

    def heatmap_pos_neg(self) -> None:
        """04 — порівняльна карта Positive vs Negative."""
        plot_pos_neg_comparison(self.layers, self.run_dir,
                                self._p("04_pos_vs_neg_heatmap.png"),
                                max_neurons=self.max_neurons)

    def heatmap_eta2(self) -> None:
        """05 — карта η² (explained variance)."""
        mat = _build_matrix(self.layers, self.run_dir, "eta2", self.max_neurons)
        if mat is None:
            print("  [viz] WARNING: eta2: no data"); return
        plot_heatmap(mat, self.layers, "η² — частка варіації, пояснена класом (pos/neg)",
                     self._p("05_eta2_heatmap.png"), cmap=_PALETTE_SEQUENTIAL, center=None)

    def heatmap_snr(self) -> None:
        """06 — карта SNR."""
        mat = _build_matrix(self.layers, self.run_dir, "snr", self.max_neurons)
        if mat is None:
            print("  [viz] WARNING: snr: no data"); return
        plot_heatmap(mat, self.layers, "SNR = |Δ| / (std_pos + std_neg)",
                     self._p("06_snr_heatmap.png"), cmap=_PALETTE_SEQUENTIAL, center=None)

    def profile_layers(self) -> None:
        """07 — профіль метрик по шарах."""
        plot_layer_profile(self.layers, self.run_dir, self._p("07_layer_profile.png"))

    def html_interactive_delta(self) -> None:
        """08 — інтерактивна карта delta (HTML)."""
        if not self.generate_html: return
        mat = _build_matrix(self.layers, self.run_dir, "delta", self.max_neurons)
        if mat is None: return
        plot_interactive_heatmap(mat, self.layers,
                                 "Δ (mean_pos − mean_neg) — інтерактивна карта",
                                 self._p("08_interactive_delta.html"))

    def html_interactive_effect(self) -> None:
        """09 — інтерактивний барплот effect_size (HTML)."""
        if not self.generate_html: return
        plot_interactive_top_neurons(self.layers, self.run_dir,
                                     self._p("09_interactive_effect.html"),
                                     top_n=self.top_n * 2)

    def summary_dashboard(self) -> None:
        """Зведений дашборд (report_summary.png)."""
        plot_summary_dashboard(self.layers, self.run_dir,
                               self._p("report_summary.png"),
                               top_n=self.top_n,
                               max_neurons=self.max_neurons)

    def run_all(self) -> str:
        """
        Генерує всі графіки. Повертає шлях до папки з виходами.
        """
        print(f"\n[viz] Generating visualizations for: {self.run_name}")
        print(f"[viz] Шари: {self.layers}\n")

        steps = [
            ("01 delta heatmap",         self.heatmap_delta),
            ("02 effect_size heatmap",   self.heatmap_effect_size),
            ("03 top neurons bar",       self.bar_top_neurons),
            ("04 pos vs neg heatmap",    self.heatmap_pos_neg),
            ("05 eta2 heatmap",          self.heatmap_eta2),
            ("06 snr heatmap",           self.heatmap_snr),
            ("07 layer profile",         self.profile_layers),
            ("08 interactive delta",     self.html_interactive_delta),
            ("09 interactive effect",    self.html_interactive_effect),
            ("summary dashboard",        self.summary_dashboard),
        ]

        for name, fn in steps:
            try:
                print(f"  [viz] >> {name}...")
                fn()
            except Exception as exc:
                warnings.warn(f"[viz] ERROR: {name} failed: {exc}")

        print(f"\n[viz] Done. All files saved to:\n  {self.out_dir}\n")
        return self.out_dir

# CLI entrypoint
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizer — генератор теплових карт та звітів для LLM-Emotion-Interpretability."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Шлях до папки з активаціями, напр. data/activations/run_20260225_0035",
    )
    parser.add_argument("--top_n", type=int, default=30, help="Кількість топ-нейронів для барплотів.")
    parser.add_argument("--max_neurons", type=int, default=512, help="Макс. нейронів у теплових картах.")
    parser.add_argument("--no_html", action="store_true", help="Не генерувати HTML (Plotly).")
    parser.add_argument(
        "--out_subdir",
        default=None,
        help="Підпапка виходів (за замовчуванням outputs/heatmaps/<run_name>).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    viz = Visualizer(
        run_dir=run_dir,
        out_subdir=args.out_subdir,
        top_n=args.top_n,
        max_neurons=args.max_neurons,
        generate_html=not args.no_html,
    )
    viz.run_all()


if __name__ == "__main__":
    main()