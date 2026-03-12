"""Inspect a run directory: export JSON and generate charts.

All outputs are saved directly inside <run-dir>/ alongside the .pt files:
  - <stem>.json           — tensor data (truncated for large tensors)
  - plot_topk.png         — top-k token probabilities per generation step
  - plot_hidden_norms.png — L2 norm of hidden states per layer
  - plot_attention.png    — attention heatmap (layer 0, all heads, prompt pass)
  - plot_logit_dist.png   — logit distribution across generation steps

Usage:
    python inspect_run.py --run-dir experiments/run_20240101_120000
    python inspect_run.py --run-dir experiments/run_20240101_120000 --no-plots
    python inspect_run.py --run-dir experiments/run_20240101_120000 --no-json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def flatten_structure(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Recursively flatten nested tuples/lists/dicts of tensors → flat dict."""
    result: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        result[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            result.update(flatten_structure(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(flatten_structure(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    return result


def sanitize(values: List[float]) -> List:
    """Replace non-finite floats with None for valid JSON."""
    return [None if not np.isfinite(v) else v for v in values]


def shape_summary(obj: Any, indent: int = 0) -> List[str]:
    lines = []
    pad = "  " * indent
    if isinstance(obj, torch.Tensor):
        arr = obj.float()
        finite = arr[torch.isfinite(arr)]
        stats = (f"min={finite.min():.4f}  max={finite.max():.4f}  mean={finite.mean():.4f}"
                 if finite.numel() > 0 else "all non-finite")
        lines.append(f"{pad}Tensor  shape={list(obj.shape)}  dtype={obj.dtype}  {stats}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(f"{pad}['{k}']")
            lines.extend(shape_summary(v, indent + 1))
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{pad}{type(obj).__name__}  len={len(obj)}")
        if len(obj) > 0:
            lines.extend(shape_summary(obj[0], indent + 1))
            if len(obj) > 1:
                lines.append(f"{pad}  ... ({len(obj)} items total, showing first)")
    else:
        lines.append(f"{pad}{type(obj).__name__}: {obj}")
    return lines


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(flat: Dict[str, torch.Tensor], run_dir: Path, stem: str) -> None:
    MAX_ELEMENTS = 10_000
    payload = {}
    for key, tensor in flat.items():
        arr = tensor.float().numpy()
        total = arr.size
        flat_data = sanitize(arr.flatten().tolist())
        if total <= MAX_ELEMENTS:
            payload[key] = {"shape": list(arr.shape), "data": flat_data}
        else:
            payload[key] = {
                "shape": list(arr.shape),
                "truncated": True,
                "total_elements": int(total),
                "data": flat_data[:MAX_ELEMENTS],
            }
    out_path = run_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  JSON → {out_path.name}")


# ── charts ────────────────────────────────────────────────────────────────────

def plot_topk(run_dir: Path, out_dir: Path) -> None:
    """Bar chart: top-k token probabilities per generation step."""
    import matplotlib.pyplot as plt

    json_path = run_dir / "top_tokens_per_step.json"
    if not json_path.exists():
        print("  plot_topk: top_tokens_per_step.json not found, skipping")
        return

    steps = json.loads(json_path.read_text(encoding="utf-8"))
    n_steps = len(steps)
    if n_steps == 0:
        return

    show = min(n_steps, 10)
    cols = 5
    rows = (show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = np.array(axes).flatten()

    for i in range(show):
        ax = axes[i]
        step = steps[i]
        topk = [(e["token"], e["logprob"]) for e in step["topk"] if e["logprob"] is not None]
        topk = topk[:10]
        tokens = [t.replace("\n", "\\n").replace(" ", "·") for t, _ in topk]
        probs = np.exp([lp for _, lp in topk])
        predicted = step["predicted_token"].replace("\n", "\\n").replace(" ", "·")
        colors = ["#e74c3c" if t == predicted else "#3498db" for t in tokens]
        ax.barh(range(len(tokens)), probs, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"Step {step['step']}  →  \"{step['predicted_token'].strip()}\"",
                     fontsize=8, pad=3)
        ax.set_xlabel("probability", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top-k token probabilities per generation step  (red = chosen)", fontsize=11)
    plt.tight_layout()
    out = out_dir / "plot_topk.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot → {out.name}")


def plot_hidden_norms(run_dir: Path, out_dir: Path) -> None:
    """Line chart: mean L2 norm of hidden states per layer."""
    import matplotlib.pyplot as plt

    pt_path = run_dir / "prompt_hidden_states.pt"
    if not pt_path.exists():
        print("  plot_hidden_norms: prompt_hidden_states.pt not found, skipping")
        return

    hidden_states = load_pt(pt_path)
    norms = []
    for layer_hs in hidden_states:
        if isinstance(layer_hs, torch.Tensor):
            norms.append(layer_hs.float().norm(dim=-1).mean().item())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(norms)), norms, marker="o", linewidth=1.8, markersize=4, color="#2ecc71")
    ax.fill_between(range(len(norms)), norms, alpha=0.15, color="#2ecc71")
    ax.set_xlabel("Layer index  (0 = embedding)", fontsize=10)
    ax.set_ylabel("Mean L2 norm", fontsize=10)
    ax.set_title("Hidden state L2 norm per layer  (prompt pass)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = out_dir / "plot_hidden_norms.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot → {out.name}")


def plot_attention(run_dir: Path, out_dir: Path) -> None:
    """Heatmap: attention weights for layer 0, all heads, prompt pass."""
    import matplotlib.pyplot as plt

    pt_path = run_dir / "prompt_attentions.pt"
    if not pt_path.exists():
        print("  plot_attention: prompt_attentions.pt not found, skipping")
        return

    attentions = load_pt(pt_path)
    layer0 = attentions[0]
    if not isinstance(layer0, torch.Tensor):
        print("  plot_attention: unexpected format, skipping")
        return

    attn = layer0[0].float().numpy()  # (n_heads, seq, seq)
    n_heads = attn.shape[0]
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.8))
    axes = np.array(axes).flatten()

    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(attn[h], aspect="auto", cmap="Blues", vmin=0, vmax=attn[h].max())
        ax.set_title(f"Head {h}", fontsize=8)
        ax.set_xlabel("key", fontsize=7)
        ax.set_ylabel("query", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(n_heads, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Attention weights — layer 0, prompt pass", fontsize=11)
    plt.tight_layout()
    out = out_dir / "plot_attention.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot → {out.name}")


def plot_logit_dist(run_dir: Path, out_dir: Path) -> None:
    """Line chart: min/mean/max of finite logits per generation step."""
    import matplotlib.pyplot as plt

    pt_path = run_dir / "generation_logits.pt"
    if not pt_path.exists():
        print("  plot_logit_dist: generation_logits.pt not found, skipping")
        return

    gen_logits = load_pt(pt_path)
    if not isinstance(gen_logits, torch.Tensor):
        gen_logits = torch.stack(gen_logits)
    gen_logits = gen_logits.float()

    means, maxs, mins = [], [], []
    for i in range(gen_logits.shape[0]):
        row = gen_logits[i]
        finite = row[torch.isfinite(row)]
        if finite.numel() == 0:
            means.append(float("nan"))
            maxs.append(float("nan"))
            mins.append(float("nan"))
        else:
            means.append(finite.mean().item())
            maxs.append(finite.max().item())
            mins.append(finite.min().item())

    x = range(len(means))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, maxs,  label="max logit",  color="#e74c3c", linewidth=1.5)
    ax.plot(x, means, label="mean logit", color="#3498db", linewidth=1.5)
    ax.plot(x, mins,  label="min logit",  color="#95a5a6", linewidth=1.0, linestyle="--")
    ax.fill_between(x, mins, maxs, alpha=0.07, color="#3498db")
    ax.set_xlabel("Generation step", fontsize=10)
    ax.set_ylabel("Logit value  (finite only)", fontsize=10)
    ax.set_title("Logit distribution across generation steps", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = out_dir / "plot_logit_dist.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot → {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect run artifacts: export JSON and charts")
    parser.add_argument("--run-dir", required=True,
                        help="Path to a run directory (e.g. experiments/run_20240101_120000)")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-json",  action="store_true", help="Skip JSON export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        return

    out_dir = run_dir / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON export ───────────────────────────────────────────────────────────
    if not args.no_json:
        pt_files = sorted(run_dir.glob("*.pt"))
        if pt_files:
            print(f"Exporting {len(pt_files)} .pt files to JSON...")
            for pt_path in pt_files:
                stem = pt_path.stem
                data = load_pt(pt_path)
                for line in shape_summary(data):
                    print("   " + line)
                flat = flatten_structure(data)
                export_json(flat, out_dir, stem)
                print()
        else:
            print("No .pt files found.")

    # ── charts ────────────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            print("Generating charts...")
            plot_topk(run_dir, out_dir)
            plot_hidden_norms(run_dir, out_dir)
            plot_attention(run_dir, out_dir)
            plot_logit_dist(run_dir, out_dir)
        except ImportError:
            print("matplotlib not installed — skipping charts.  Run: pip install matplotlib")

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()