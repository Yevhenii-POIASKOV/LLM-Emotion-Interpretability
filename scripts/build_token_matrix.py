"""Stack all captured activations into a single token matrix.

For each token in the prompt, collects:
  - Hidden state from every layer           → n_layers+1 × hidden_dim
  - Q, K, V vectors from every layer        → 3 × n_layers × hidden_dim
  - MLP pre-activation from every layer     → n_layers × mlp_dim
  - MLP post-activation from every layer    → n_layers × mlp_dim
  - Mean attention received per layer       → n_layers  (scalar, fixed size)

Result: matrix of shape  (seq_len, total_features)
Saved as:
  <run-dir>/exported/stacked_tokens.pt   — torch tensor
  <run-dir>/exported/stacked_tokens.npy  — numpy array
  <run-dir>/exported/stacked_tokens_meta.json — feature index map

Usage:
    python build_token_matrix.py --run-dir experiments/run_20240101_120000
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── loaders ───────────────────────────────────────────────────────────────────

def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


# ── builder ───────────────────────────────────────────────────────────────────

def build_token_matrix(run_dir: Path):
    """
    Returns:
        matrix  — torch.Tensor  (seq_len, total_features)
        meta    — dict describing which feature indices correspond to what
    """

    # ── load artifacts ───────────────────────────────────────────────────────
    hs_path   = run_dir / "prompt_hidden_states.pt"
    hook_path = run_dir / "prompt_hooks.pt"
    att_path  = run_dir / "prompt_attentions.pt"

    for p in (hs_path, hook_path, att_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    hidden_states = load_pt(hs_path)   # tuple[ (1, seq, hidden) ]  len = n_layers+1
    hooks         = load_pt(hook_path)  # dict key → list[tensor]
    attentions    = load_pt(att_path)   # tuple[ (1, heads, seq, seq) ]  len = n_layers

    # ── basic dimensions ─────────────────────────────────────────────────────
    seq_len    = hidden_states[0].shape[1]   # number of prompt tokens
    hidden_dim = hidden_states[0].shape[2]
    n_hs_layers = len(hidden_states)         # includes embedding (layer 0)

    # Detect number of transformer blocks from hooks
    block_indices = sorted({
        int(k.split(".")[1])
        for k in hooks
        if k.startswith("block.")
    })
    n_blocks = len(block_indices)

    print(f"  tokens       : {seq_len}")
    print(f"  hidden dim   : {hidden_dim}")
    print(f"  hs layers    : {n_hs_layers}  (embedding + {n_hs_layers-1} transformer layers)")
    print(f"  blocks (hooks): {n_blocks}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def hook_tensor(key: str) -> Optional[torch.Tensor]:
        """Return the first tensor stored under a hook key, or None."""
        val = hooks.get(key)
        if val is None or len(val) == 0:
            return None
        t = val[0]  # first forward pass recording
        if isinstance(t, torch.Tensor):
            return t.float()
        return None

    def hook_slice(key: str, token_idx: int) -> Optional[torch.Tensor]:
        """Return the feature vector for one token from a hook tensor."""
        t = hook_tensor(key)
        if t is None:
            return None
        # shape: (batch, seq, features)
        if t.ndim == 3:
            return t[0, token_idx, :]
        # shape: (seq, features)
        if t.ndim == 2:
            return t[token_idx, :]
        return None

    # ── detect MLP dim from first available mlp_pre_act ──────────────────────
    mlp_dim = 0
    for b in block_indices:
        t = hook_tensor(f"block.{b}.mlp_pre_act")
        if t is not None:
            mlp_dim = t.shape[-1]
            break
    print(f"  mlp dim      : {mlp_dim if mlp_dim else 'not captured'}")

    # ── build feature index map ───────────────────────────────────────────────
    # We record (name, start_idx, end_idx) for each feature group
    meta_segments: List[Dict] = []
    cursor = 0

    def register(name: str, size: int) -> int:
        nonlocal cursor
        meta_segments.append({"name": name, "start": cursor, "end": cursor + size, "size": size})
        cursor += size
        return size

    # Hidden states: one entry per layer (including embedding)
    for l in range(n_hs_layers):
        register(f"hidden_state.layer_{l}", hidden_dim)

    # Q, K, V per block
    for b in block_indices:
        for proj in ("attn_q", "attn_k", "attn_v"):
            t = hook_tensor(f"block.{b}.{proj}")
            dim = t.shape[-1] if t is not None else hidden_dim
            register(f"block_{b}.{proj}", dim)

    # MLP pre / post activation per block
    if mlp_dim > 0:
        for b in block_indices:
            register(f"block_{b}.mlp_pre_act",  mlp_dim)
            register(f"block_{b}.mlp_post_act", mlp_dim)

    # Mean attention received per layer (scalar → 1 feature per layer)
    for l in range(len(attentions)):
        register(f"attn_received_mean.layer_{l}", 1)

    total_features = cursor
    print(f"  total features: {total_features}")

    # ── fill matrix ───────────────────────────────────────────────────────────
    matrix = torch.zeros(seq_len, total_features, dtype=torch.float32)

    for t_idx in range(seq_len):
        vec = torch.zeros(total_features)
        pos = 0

        # Hidden states
        for l, hs in enumerate(hidden_states):
            if isinstance(hs, torch.Tensor):
                vec[pos: pos + hidden_dim] = hs[0, t_idx, :].float()
            pos += hidden_dim

        # Q, K, V
        for b in block_indices:
            for proj in ("attn_q", "attn_k", "attn_v"):
                sl = hook_slice(f"block.{b}.{proj}", t_idx)
                t0 = hook_tensor(f"block.{b}.{proj}")
                dim = t0.shape[-1] if t0 is not None else hidden_dim
                if sl is not None:
                    vec[pos: pos + dim] = sl
                pos += dim

        # MLP pre / post
        if mlp_dim > 0:
            for b in block_indices:
                for hook_name in ("mlp_pre_act", "mlp_post_act"):
                    sl = hook_slice(f"block.{b}.{hook_name}", t_idx)
                    if sl is not None:
                        vec[pos: pos + mlp_dim] = sl
                    pos += mlp_dim

        # Mean attention received by this token (averaged over heads and queries)
        for l, att_layer in enumerate(attentions):
            if isinstance(att_layer, torch.Tensor):
                # att_layer: (1, heads, seq, seq) — axis -1 = key token
                # column t_idx = how much attention token t_idx receives from others
                received = att_layer[0, :, :, t_idx].mean().item()  # mean over heads & queries
                vec[pos] = received
            pos += 1

        matrix[t_idx] = vec

    return matrix, {"seq_len": seq_len, "total_features": total_features, "segments": meta_segments}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-token feature matrix from run artifacts")
    parser.add_argument("--run-dir", required=True,
                        help="Path to run directory (e.g. experiments/run_20240101_120000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return

    out_dir = run_dir / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building token matrix from: {run_dir}")
    matrix, meta = build_token_matrix(run_dir)

    # Save
    pt_path  = out_dir / "stacked_tokens.pt"
    npy_path = out_dir / "stacked_tokens.npy"
    meta_path = out_dir / "stacked_tokens_meta.json"

    torch.save(matrix, pt_path)
    np.save(npy_path, matrix.numpy())
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMatrix shape : {list(matrix.shape)}  (tokens × features)")
    print(f"Saved:")
    print(f"  {pt_path.name}")
    print(f"  {npy_path.name}")
    print(f"  {meta_path.name}")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
