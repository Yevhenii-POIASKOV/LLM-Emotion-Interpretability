"""
analyzer.py

Перетворює збережені активації (mlp_layer_*.npy) + labels.npy у статистику:
- mean_pos / mean_neg для кожного шару і нейрона
- delta = mean_pos - mean_neg
- метрики "чутливості" нейронів до емоції:
  - abs_delta
  - effect_size (Cohen's d)  <-- основна метрика для "найбільш чутливі"
  - eta2 (частка варіації, пояснена класом pos/neg)
  - snr (|delta| / (std_pos + std_neg))

ДОДАНО ПЕРЕВІРКИ / "якісність" для відбору (пункт 4):
- Bootstrap CI для delta для топ-кандидатів (за метрикою відбору)
  -> визначає "стабільні" нейрони: 95% CI для delta НЕ перетинає 0

Це робить відбір "найбільш чутливих" більш коректним (не просто максимальний observed effect),
але без важких залежностей (тільки numpy).

Очікувана структура даних:
data/activations/run_YYYYMMDD_HHMM/
  mlp_layer_0.npy ... mlp_layer_11.npy   shape: (N, d_mlp)
  labels.npy                             shape: (N,)  values: 0/1 (neg/pos)

Запуск:
  conda activate llm_project
  cd <repo_root>
  python -m src.analyzer --run_dir data/activations/run_20260225_0035

Простіший запуск як файл:
  cd <repo_root>/src
  python analyzer.py --run_dir ../data/activations/run_20260225_0035

Керування selection:
  # Найбільш чутливі (емоційні) нейрони: global top-50 за effect_size + bootstrap-check
  python -m src.analyzer --run_dir data/activations/run_... --select_global --select_top_n 50

Bootstrap перевірка (дефолт увімкнена):
  --bootstrap_enable
  --bootstrap_candidates 200   (скільки кандидатів перевіряти)
  --bootstrap_iters 1000       (кількість bootstrap ітерацій)
  --bootstrap_alpha 0.05       (для 95% CI)

Порада по швидкості:
- bootstrap по всіх 12*3072 нейронах буде довго.
- тут bootstrap застосовується лише до top-N кандидатів (за score) у selection.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class LayerStats:
    layer: int
    n_samples: int
    n_pos: int
    n_neg: int

    mean_pos: np.ndarray  # (d_mlp,)
    mean_neg: np.ndarray  # (d_mlp,)
    delta: np.ndarray     # (d_mlp,)

    std_all: np.ndarray   # (d_mlp,)
    std_pos: np.ndarray   # (d_mlp,)
    std_neg: np.ndarray   # (d_mlp,)

    abs_delta: np.ndarray     # (d_mlp,)
    effect_size: np.ndarray   # Cohen's d, (d_mlp,)

    between_var: np.ndarray   # (d_mlp,) between-class variance component
    within_var: np.ndarray    # (d_mlp,) pooled within-class variance
    eta2: np.ndarray          # (d_mlp,) explained variance ratio
    snr: np.ndarray           # (d_mlp,) |delta| / (std_pos + std_neg)


@dataclass
class BootstrapResult:
    layer: int
    neuron: int
    delta_hat: float
    ci_low: float
    ci_high: float
    alpha: float
    iters: int
    stable_nonzero: bool  # True if CI does NOT cross 0


# -----------------------------
# Helpers
# -----------------------------
def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _guess_repo_root_from_src_file() -> str:
    # src/analyzer.py -> repo_root = parent of src
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_labels(run_dir: str) -> np.ndarray:
    labels_path = os.path.join(run_dir, "labels.npy")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.npy not found in run_dir: {run_dir}")

    labels = np.load(labels_path)
    labels = labels.astype(np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels.npy is empty")

    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"labels.npy must contain only 0/1. Got: {sorted(unique)}")
    return labels


def _find_layer_files(run_dir: str) -> List[Tuple[int, str]]:
    """Returns list of (layer_idx, path) sorted by layer_idx."""
    files: List[Tuple[int, str]] = []
    for name in os.listdir(run_dir):
        if name.startswith("mlp_layer_") and name.endswith(".npy"):
            try:
                layer_str = name[len("mlp_layer_") : -len(".npy")]
                layer = int(layer_str)
            except ValueError:
                continue
            files.append((layer, os.path.join(run_dir, name)))

    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(f"No mlp_layer_*.npy files found in {run_dir}")
    return files


def _cohens_d(
    mean_pos: np.ndarray,
    mean_neg: np.ndarray,
    std_pos: np.ndarray,
    std_neg: np.ndarray,
    n_pos: int,
    n_neg: int,
) -> np.ndarray:
    """Cohen's d per-neuron (pooled std)."""
    eps = 1e-8
    denom = max((n_pos + n_neg - 2), 1)
    pooled_var = ((n_pos - 1) * (std_pos**2) + (n_neg - 1) * (std_neg**2)) / denom
    pooled_std = np.sqrt(np.maximum(pooled_var, 0.0)) + eps
    return (mean_pos - mean_neg) / pooled_std


# -----------------------------
# Core computations
# -----------------------------
def compute_layer_stats(acts: np.ndarray, labels: np.ndarray, layer: int) -> LayerStats:
    """
    acts: (N, d_mlp)
    labels: (N,) values 0/1
    """
    if acts.ndim != 2:
        raise ValueError(f"Layer {layer}: expected 2D acts, got shape={acts.shape}")
    if acts.shape[0] != labels.shape[0]:
        raise ValueError(f"Layer {layer}: acts N={acts.shape[0]} != labels N={labels.shape[0]}")

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Layer {layer}: need both pos and neg samples. n_pos={n_pos}, n_neg={n_neg}")
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Layer {layer}: need >=2 samples per class for ddof=1 std. n_pos={n_pos}, n_neg={n_neg}")
    
    acts_pos = acts[pos_mask]
    acts_neg = acts[neg_mask]

    mean_pos = acts_pos.mean(axis=0)
    mean_neg = acts_neg.mean(axis=0)
    delta = mean_pos - mean_neg

    std_all = acts.std(axis=0, ddof=1)
    std_pos = acts_pos.std(axis=0, ddof=1)
    std_neg = acts_neg.std(axis=0, ddof=1)

    abs_delta = np.abs(delta)
    effect_size = _cohens_d(mean_pos, mean_neg, std_pos, std_neg, n_pos=n_pos, n_neg=n_neg)

    # Variability decomposition: between vs within (2-class)
    n = n_pos + n_neg
    between_var = (n_pos * n_neg) / n * (delta**2)
    within_var = ((n_pos - 1) * (std_pos**2) + (n_neg - 1) * (std_neg**2)) / max((n - 2), 1)

    eps = 1e-12
    eta2 = between_var / (between_var + within_var + eps)

    # simple signal-to-noise ratio
    snr_eps = 1e-8
    snr_denom = np.maximum(std_pos + std_neg, snr_eps)
    snr = abs_delta / snr_denom

    return LayerStats(
        layer=layer,
        n_samples=int(acts.shape[0]),
        n_pos=n_pos,
        n_neg=n_neg,
        mean_pos=mean_pos,
        mean_neg=mean_neg,
        delta=delta,
        std_all=std_all,
        std_pos=std_pos,
        std_neg=std_neg,
        abs_delta=abs_delta,
        effect_size=effect_size,
        between_var=between_var,
        within_var=within_var,
        eta2=eta2,
        snr=snr,
    )


def _record_neuron(st: LayerStats, neuron_idx: int, score: str) -> Dict:
    i = int(neuron_idx)
    return {
        "layer": int(st.layer),
        "neuron": i,
        "score_name": score,
        "score": float(getattr(st, score)[i]),
        "delta": float(st.delta[i]),
        "abs_delta": float(st.abs_delta[i]),
        "std_all": float(st.std_all[i]),
        "effect_size": float(st.effect_size[i]),
        "eta2": float(st.eta2[i]),
        "snr": float(st.snr[i]),
        "mean_pos": float(st.mean_pos[i]),
        "mean_neg": float(st.mean_neg[i]),
    }


def top_neurons(layer_stats: LayerStats, k: int = 20, score: str = "effect_size") -> List[Dict]:
    """
    score: abs_delta | std_all | effect_size | eta2 | snr
    Returns list of dicts with neuron_idx + metrics.
    """
    if score not in {"abs_delta", "std_all", "effect_size", "eta2", "snr"}:
        raise ValueError("score must be one of: abs_delta, std_all, effect_size, eta2, snr")

    arr = getattr(layer_stats, score)
    idx = np.argsort(arr)[::-1][:k]
    return [_record_neuron(layer_stats, int(i), score=score) for i in idx]


# -----------------------------
# Selection: "library" functions
# -----------------------------
def select_neurons(
    layer_stats_map: Dict[int, LayerStats],
    *,
    mode: str = "top_n",            # "top_n" або "threshold"
    score: str = "effect_size",     # DEFAULT: "most sensitive" to emotion
    top_n: int = 50,                # for mode="top_n"
    threshold: float = 0.0,         # for mode="threshold"
    per_layer: bool = True,         # True: per-layer, False: global list
) -> Dict:
    """
    Повертає структуру з обраними нейронами.
    """
    if score not in {"abs_delta", "std_all", "effect_size", "eta2", "snr"}:
        raise ValueError("score must be one of: abs_delta, std_all, effect_size, eta2, snr")
    if mode not in {"top_n", "threshold"}:
        raise ValueError("mode must be 'top_n' or 'threshold'")

    layers = sorted(layer_stats_map.keys())

    if per_layer:
        selected_by_layer: Dict[str, List[Dict]] = {}
        for layer in layers:
            st = layer_stats_map[layer]
            scores = getattr(st, score)

            if mode == "top_n":
                idx = np.argsort(scores)[::-1][:top_n]
            else:
                idx = np.where(scores >= threshold)[0]
                idx = idx[np.argsort(scores[idx])[::-1]]

            selected_by_layer[str(layer)] = [_record_neuron(st, int(i), score=score) for i in idx]

        return {
            "mode": mode,
            "score": score,
            "top_n": top_n,
            "threshold": threshold,
            "per_layer": True,
            "selected_by_layer": selected_by_layer,
        }

    # global list
    all_records: List[Dict] = []
    for layer in layers:
        st = layer_stats_map[layer]
        scores = getattr(st, score)

        if mode == "top_n":
            idx = np.arange(scores.shape[0])
        else:
            idx = np.where(scores >= threshold)[0]

        for i in idx:
            all_records.append(_record_neuron(st, int(i), score=score))

    all_records.sort(key=lambda r: r["score"], reverse=True)

    if mode == "top_n":
        all_records = all_records[:top_n]

    return {
        "mode": mode,
        "score": score,
        "top_n": top_n,
        "threshold": threshold,
        "per_layer": False,
        "selected": all_records,
    }


def save_selected_neurons(
    repo_root: str,
    run_dir: str,
    selection: Dict,
    out_subdir: str = "outputs/reports",
) -> str:
    """
    Зберігає selection у JSON:
      outputs/reports/selected_neurons_<run_name>.json
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    out_dir = os.path.join(repo_root, out_subdir)
    _safe_makedirs(out_dir)

    path = os.path.join(out_dir, f"selected_neurons_{run_name}.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "run_name": run_name,
        **selection,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


# -----------------------------
# Bootstrap checks (quality)
# -----------------------------
def bootstrap_delta_ci_for_neuron(
    acts: np.ndarray,
    labels: np.ndarray,
    neuron_idx: int,
    *,
    iters: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for delta = mean_pos - mean_neg for ONE neuron.

    Returns: (delta_hat, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = acts[:, int(neuron_idx)]
    x_pos = x[labels == 1]
    x_neg = x[labels == 0]
    n_pos = x_pos.shape[0]
    n_neg = x_neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both pos and neg samples for bootstrap.")

    delta_hat = float(x_pos.mean() - x_neg.mean())

    # Sample indices with replacement
    pos_idx = rng.integers(0, n_pos, size=(iters, n_pos))
    neg_idx = rng.integers(0, n_neg, size=(iters, n_neg))

    pos_means = x_pos[pos_idx].mean(axis=1)
    neg_means = x_neg[neg_idx].mean(axis=1)
    deltas = pos_means - neg_means

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return delta_hat, lo, hi


def bootstrap_check_selection(
    run_dir: str,
    labels: np.ndarray,
    layer_files: List[Tuple[int, str]],
    selection: Dict,
    *,
    score: str,
    candidates: int = 200,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict:
    """
    Adds bootstrap CI results for top candidate neurons.

    Strategy:
    - Build a flat candidate list (layer, neuron, score) from selection
      - if per_layer=True: take first `candidates` across layers in round-robin style
      - if per_layer=False: take first `candidates` from the global list
    - For each candidate, compute bootstrap CI for delta on that neuron.

    Returns dict with:
      - bootstrap_enabled
      - bootstrap_params
      - bootstrap_results (list)
      - stable_count
    """
    rng = np.random.default_rng(seed)

    # Build quick map layer->path
    layer_to_path = {layer: path for layer, path in layer_files}

    # Extract candidate list
    cand_list: List[Tuple[int, int, float]] = []  # (layer, neuron, score_value)

    if selection.get("per_layer", False):
        selected_by_layer = selection["selected_by_layer"]
        # round-robin over layers to avoid taking all from layer 0 first
        layers = sorted(int(k) for k in selected_by_layer.keys())
        pointers = {L: 0 for L in layers}
        while len(cand_list) < candidates:
            progressed = False
            for L in layers:
                arr = selected_by_layer[str(L)]
                p = pointers[L]
                if p < len(arr):
                    rec = arr[p]
                    cand_list.append((int(rec["layer"]), int(rec["neuron"]), float(rec["score"])))
                    pointers[L] += 1
                    progressed = True
                    if len(cand_list) >= candidates:
                        break
            if not progressed:
                break
    else:
        selected = selection["selected"]
        for rec in selected[:candidates]:
            cand_list.append((int(rec["layer"]), int(rec["neuron"]), float(rec["score"])))

    results: List[Dict] = []
    stable = 0

    # Cache activations per layer (load once)
    layer_cache: Dict[int, np.ndarray] = {}

    for layer, neuron, _score_val in cand_list:
        if layer not in layer_cache:
            acts = np.load(layer_to_path[layer])
            layer_cache[layer] = acts
        else:
            acts = layer_cache[layer]

        delta_hat, lo, hi = bootstrap_delta_ci_for_neuron(
            acts, labels, neuron, iters=iters, alpha=alpha, rng=rng
        )
        stable_nonzero = not (lo <= 0.0 <= hi)
        if stable_nonzero:
            stable += 1

        br = BootstrapResult(
            layer=layer,
            neuron=neuron,
            delta_hat=delta_hat,
            ci_low=lo,
            ci_high=hi,
            alpha=alpha,
            iters=iters,
            stable_nonzero=stable_nonzero,
        )
        results.append(br.__dict__)

    return {
        "bootstrap_enabled": True,
        "bootstrap_params": {
            "score": score,
            "candidates": candidates,
            "iters": iters,
            "alpha": alpha,
            "seed": seed,
        },
        "bootstrap_results": results,
        "stable_count": stable,
        "checked_count": len(results),
    }


# -----------------------------
# Reporting outputs
# -----------------------------
def save_outputs(
    repo_root: str,
    run_dir: str,
    layer_stats_map: Dict[int, LayerStats],
    topk: int = 50,
    out_subdir: str = "outputs/reports",
) -> str:
    """
    Saves:
    - outputs/reports/analyzer_<run_name>.json (summary + top neurons)
    - outputs/reports/analyzer_<run_name>_layer_<L>.npz (arrays)
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    out_dir = os.path.join(repo_root, out_subdir)
    _safe_makedirs(out_dir)

    # Save arrays per layer to .npz
    for layer, st in layer_stats_map.items():
        npz_path = os.path.join(out_dir, f"analyzer_{run_name}_layer_{layer}.npz")
        np.savez_compressed(
            npz_path,
            mean_pos=st.mean_pos,
            mean_neg=st.mean_neg,
            delta=st.delta,
            abs_delta=st.abs_delta,
            std_all=st.std_all,
            std_pos=st.std_pos,
            std_neg=st.std_neg,
            effect_size=st.effect_size,
            between_var=st.between_var,
            within_var=st.within_var,
            eta2=st.eta2,
            snr=st.snr,
            n_samples=st.n_samples,
            n_pos=st.n_pos,
            n_neg=st.n_neg,
        )

    # JSON summary (tops)
    tops: Dict[str, List[Dict]] = {
        "top_by_abs_delta": [],
        "top_by_std_all": [],
        "top_by_effect_size": [],
        "top_by_eta2": [],
        "top_by_snr": [],
    }

    for layer in sorted(layer_stats_map.keys()):
        st = layer_stats_map[layer]
        tops["top_by_abs_delta"].extend(top_neurons(st, k=topk, score="abs_delta"))
        tops["top_by_std_all"].extend(top_neurons(st, k=topk, score="std_all"))
        tops["top_by_effect_size"].extend(top_neurons(st, k=topk, score="effect_size"))
        tops["top_by_eta2"].extend(top_neurons(st, k=topk, score="eta2"))
        tops["top_by_snr"].extend(top_neurons(st, k=topk, score="snr"))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "run_name": run_name,
        "layers": sorted(layer_stats_map.keys()),
        "topk_per_layer": topk,
    }
    # Merge tops into summary while guarding against key collisions
    collision_keys = set(summary).intersection(tops)
    if collision_keys:
        raise ValueError(f"Key collision while building analyzer summary: {sorted(collision_keys)}")
    summary.update(tops)

    json_path = os.path.join(out_dir, f"analyzer_{run_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_dir


# -----------------------------
# CLI entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved GPT-2 MLP activations (pos vs neg).")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to a run folder, e.g. data/activations/run_20260225_0035",
    )

    # reporting
    parser.add_argument("--topk", type=int, default=50, help="Top-K neurons per layer to report in analyzer_<run>.json.")

    # selection (defaults set to "most sensitive")
    parser.add_argument("--select_mode", choices=["top_n", "threshold"], default="top_n")
    parser.add_argument(
        "--select_score",
        choices=["abs_delta", "std_all", "effect_size", "eta2", "snr"],
        default="effect_size",  # <-- default = "most sensitive"
        help="Metric used to identify 'most sensitive' neurons (default: effect_size).",
    )
    parser.add_argument("--select_top_n", type=int, default=50)
    parser.add_argument("--select_threshold", type=float, default=0.0)
    parser.add_argument("--select_per_layer", action="store_true", help="Save selection per-layer.")
    parser.add_argument("--select_global", action="store_true", help="Save selection as one global list (overrides --select_per_layer).")

    # bootstrap checks
    parser.add_argument("--bootstrap_enable", action="store_true", help="Enable bootstrap CI check for delta on top candidates.")
    parser.add_argument("--bootstrap_candidates", type=int, default=200, help="How many selected candidates to bootstrap-check.")
    parser.add_argument("--bootstrap_iters", type=int, default=1000, help="Bootstrap iterations per neuron.")
    parser.add_argument("--bootstrap_alpha", type=float, default=0.05, help="Alpha for CI (0.05 -> 95% CI).")
    parser.add_argument("--bootstrap_seed", type=int, default=0, help="Random seed for bootstrap reproducibility.")

    args = parser.parse_args()

    repo_root = _guess_repo_root_from_src_file()

    # robust absolute path from current working directory
    run_dir = os.path.abspath(args.run_dir)

    labels = _load_labels(run_dir)
    layer_files = _find_layer_files(run_dir)

    print(f"[analyzer] run_dir: {run_dir}")
    print(f"[analyzer] samples: {labels.shape[0]} (pos={int((labels==1).sum())}, neg={int((labels==0).sum())})")
    print(f"[analyzer] found {len(layer_files)} layer files")

    stats_map: Dict[int, LayerStats] = {}
    for layer, path in layer_files:
        acts = np.load(path)
        st = compute_layer_stats(acts, labels, layer=layer)
        stats_map[layer] = st

        print(
            f"[analyzer] layer {layer:02d}: acts={acts.shape}  "
            f"mean(|delta|)={float(st.abs_delta.mean()):.6f}  "
            f"max(|delta|)={float(st.abs_delta.max()):.6f}  "
            f"max(|d|)={float(np.abs(st.effect_size).max()):.6f}  "
            f"max(eta2)={float(st.eta2.max()):.6f}"
        )

    out_dir = save_outputs(repo_root=repo_root, run_dir=run_dir, layer_stats_map=stats_map, topk=args.topk)
    print(f"[analyzer] saved reports to: {out_dir}")

    # selection flags: default per-layer unless --select_global is set
    per_layer = True
    if args.select_global:
        per_layer = False
    elif args.select_per_layer:
        per_layer = True

    selection = select_neurons(
        stats_map,
        mode=args.select_mode,
        score=args.select_score,
        top_n=args.select_top_n,
        threshold=args.select_threshold,
        per_layer=per_layer,
    )

    selected_path = save_selected_neurons(repo_root=repo_root, run_dir=run_dir, selection=selection)
    print(f"[analyzer] saved selected neurons to: {selected_path}")
    print(f"[analyzer] selection metric: {args.select_score} (mode={args.select_mode}, per_layer={per_layer})")

    # bootstrap check (optional; off by default to keep runtime short)
    if args.bootstrap_enable:
        print(
            f"[analyzer] bootstrap enabled: candidates={args.bootstrap_candidates}, "
            f"iters={args.bootstrap_iters}, alpha={args.bootstrap_alpha}, seed={args.bootstrap_seed}"
        )
        boot = bootstrap_check_selection(
            run_dir=run_dir,
            labels=labels,
            layer_files=layer_files,
            selection=selection,
            score=args.select_score,
            candidates=args.bootstrap_candidates,
            iters=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
            seed=args.bootstrap_seed,
        )

        # save bootstrap results as a separate JSON to avoid bloating selected_neurons
        run_name = os.path.basename(os.path.normpath(run_dir))
        boot_path = os.path.join(repo_root, "outputs", "reports", f"bootstrap_{run_name}.json")
        _safe_makedirs(os.path.dirname(boot_path))
        with open(boot_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": run_dir,
                    "run_name": run_name,
                    "bootstrap_data": boot,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[analyzer] bootstrap checked: {boot['checked_count']} candidates; stable_nonzero={boot['stable_count']}")
        print(f"[analyzer] saved bootstrap report to: {boot_path}")


if __name__ == "__main__":
    main()