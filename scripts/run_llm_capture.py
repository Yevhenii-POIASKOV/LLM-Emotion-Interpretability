"""Run a local LLM and persist raw internals for a prompt."""
import argparse
import datetime as dt
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TensorNest = Union[torch.Tensor, Sequence[torch.Tensor], Sequence[Sequence[torch.Tensor]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture hidden states, attentions, logits, and top tokens from a local LLM")
    parser.add_argument("--prompt", default="The meaning of life is", help="Text prompt to feed into the model (default in code if not provided)")
    parser.add_argument("--model", default="gpt2", help="Model identifier (e.g., 'gpt2' or 'meta-llama/Llama-3.2-1B')")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k tokens to keep per step")
    parser.add_argument("--output-dir", default="experiments", help="Base directory to store run outputs")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow execution of custom model code from hub")
    return parser.parse_args()


def move_nested_to_cpu(nested: TensorNest) -> TensorNest:
    """Recursively detach tensors and move them to CPU."""
    if isinstance(nested, torch.Tensor):
        return nested.detach().cpu()
    if isinstance(nested, (list, tuple)):
        return type(nested)(move_nested_to_cpu(x) for x in nested)
    return nested


class ActivationRecorder:
    def __init__(self) -> None:
        self.data: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        # Some modules (e.g. transformer blocks) return tuples; take the first element
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            return
        self.data[key].append(tensor.detach().cpu())

    def to_dict(self) -> Dict[str, List[torch.Tensor]]:
        return dict(self.data)


def register_gpt2_hooks(model: AutoModelForCausalLM, get_rec: Callable[[], ActivationRecorder]) -> List:
    """Attach forward hooks for GPT-2 style modules to capture internal activations."""
    handles = []

    def add(name: str, tensor: torch.Tensor) -> None:
        rec = get_rec()
        if rec is not None:
            rec.add(name, tensor)

    # Embeddings
    if hasattr(model, "transformer"):
        tr = model.transformer
        if hasattr(tr, "wte"):
            handles.append(tr.wte.register_forward_hook(lambda m, inp, out: add("embed_token", out)))
        if hasattr(tr, "wpe"):
            handles.append(tr.wpe.register_forward_hook(lambda m, inp, out: add("embed_position", out)))

        # Blocks
        if hasattr(tr, "h"):
            for idx, block in enumerate(tr.h):
                name = f"block.{idx}"

                # ln_1 input/output
                if hasattr(block, "ln_1"):
                    handles.append(block.ln_1.register_forward_pre_hook(lambda m, inp, n=name: add(f"{n}.ln1_in", inp[0])))
                    handles.append(block.ln_1.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.ln1_out", out)))

                # Q, K, V — split output of c_attn (GPT-2 fuses QKV into one projection)
                # c_attn output shape: (batch, seq, 3 * embed_dim) → split into equal thirds
                if hasattr(block, "attn") and hasattr(block.attn, "c_attn"):
                    def _qkv_hook(m, inp, out, n=name):
                        embed_dim = out.shape[-1] // 3
                        q, k, v = out.split(embed_dim, dim=-1)
                        add(f"{n}.attn_q", q)
                        add(f"{n}.attn_k", k)
                        add(f"{n}.attn_v", v)
                    handles.append(block.attn.c_attn.register_forward_hook(_qkv_hook))

                # attention output (pre-residual add)
                if hasattr(block, "attn"):
                    handles.append(block.attn.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.attn_out", out[0] if isinstance(out, tuple) else out)))

                # ln_2 input/output
                if hasattr(block, "ln_2"):
                    handles.append(block.ln_2.register_forward_pre_hook(lambda m, inp, n=name: add(f"{n}.ln2_in", inp[0])))
                    handles.append(block.ln_2.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.ln2_out", out)))

                # MLP pre-activation: output of c_fc before GELU
                if hasattr(block, "mlp") and hasattr(block.mlp, "c_fc"):
                    handles.append(block.mlp.c_fc.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.mlp_pre_act", out)))

                # MLP post-activation: output of GELU before c_proj
                if hasattr(block, "mlp") and hasattr(block.mlp, "act"):
                    handles.append(block.mlp.act.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.mlp_post_act", out)))

                # mlp output (pre-residual add)
                if hasattr(block, "mlp"):
                    handles.append(block.mlp.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.mlp_out", out)))

                # block output (post both residual adds)
                handles.append(block.register_forward_hook(lambda m, inp, out, n=name: add(f"{n}.block_out", out)))

        # Final layernorm
        if hasattr(tr, "ln_f"):
            handles.append(tr.ln_f.register_forward_hook(lambda m, inp, out: add("ln_f_out", out)))

    return handles


def topk_per_step(
    scores: List[torch.Tensor],
    tokenizer: AutoTokenizer,
    prompt_len: int,
    generated_ids: torch.Tensor,
    k: int,
) -> List[Dict]:
    """Collect top-k logprobs for every generation step."""
    per_step: List[Dict] = []
    for i, step_scores in enumerate(scores):
        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        top_log_probs, top_indices = torch.topk(log_probs, k=k)
        predicted_token_id = int(generated_ids[prompt_len + i])
        per_step.append(
            {
                "step": i,
                "predicted_token_id": predicted_token_id,
                "predicted_token": tokenizer.decode([predicted_token_id]),
                "topk": [
                    {
                        "token": tokenizer.decode([idx]),
                        "token_id": int(idx),
                        "logprob": float(lp) if lp != float("-inf") else None,
                    }
                    for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist())
                ],
            }
        )
    return per_step


def save_run_artifacts(
    run_dir: Path,
    generated_text: str,
    prompt_hidden_states: TensorNest,
    prompt_attentions: TensorNest,
    prompt_logits: torch.Tensor,
    prompt_hooks: Dict[str, List[torch.Tensor]],
    fullseq_hooks: Dict[str, List[torch.Tensor]],
    gen_hidden_states: TensorNest,
    gen_attentions: TensorNest,
    gen_logits: torch.Tensor,
    top_tokens_first_step: Dict,
    top_tokens_all_steps: List[Dict],
    metadata: Dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "generated.txt").write_text(generated_text, encoding="utf-8")
    torch.save(prompt_hidden_states, run_dir / "prompt_hidden_states.pt")
    torch.save(prompt_attentions, run_dir / "prompt_attentions.pt")
    torch.save(prompt_logits, run_dir / "prompt_logits.pt")
    torch.save(prompt_hooks, run_dir / "prompt_hooks.pt")
    torch.save(fullseq_hooks, run_dir / "full_sequence_hooks.pt")
    torch.save(gen_hidden_states, run_dir / "generation_hidden_states.pt")
    torch.save(gen_attentions, run_dir / "generation_attentions.pt")
    torch.save(gen_logits, run_dir / "generation_logits.pt")
    (run_dir / "top_tokens_first_step.json").write_text(
        json.dumps(top_tokens_first_step, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "top_tokens_per_step.json").write_text(
        json.dumps(top_tokens_all_steps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path(args.output_dir)
    timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        output_hidden_states=True,
        output_attentions=True,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.to(device)
    model.eval()

    prompt_hooks = ActivationRecorder()
    fullseq_hooks = ActivationRecorder()

    active_rec: Dict[str, ActivationRecorder] = {"rec": None}
    def get_rec() -> ActivationRecorder:
        return active_rec["rec"]

    hook_handles = register_gpt2_hooks(model, get_rec)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        active_rec["rec"] = prompt_hooks
        prompt_forward = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        active_rec["rec"] = None

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            output_scores=True,
            use_cache=True,
        )

    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Full-sequence forward pass (prompt + generated) to capture deep activations with hooks
    full_seq_inputs = {"input_ids": generated_ids.unsqueeze(0).to(device)}
    if "attention_mask" in encoded:
        full_seq_inputs["attention_mask"] = torch.ones_like(full_seq_inputs["input_ids"], device=device)
    with torch.no_grad():
        active_rec["rec"] = fullseq_hooks
        _ = model(
            **full_seq_inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        active_rec["rec"] = None

    prompt_hidden_states = move_nested_to_cpu(prompt_forward.hidden_states)
    prompt_attentions = move_nested_to_cpu(prompt_forward.attentions)
    prompt_logits = prompt_forward.logits.detach().cpu()

    gen_hidden_states = move_nested_to_cpu(outputs.hidden_states)
    gen_attentions = move_nested_to_cpu(outputs.attentions)

    gen_logits = torch.stack(outputs.scores).squeeze(1).detach().cpu()

    first_step_scores = outputs.scores[0][0]
    log_probs = torch.log_softmax(first_step_scores, dim=-1)
    top_log_probs, top_indices = torch.topk(log_probs, k=args.top_k)
    top_tokens_first_step = {
        "topk": [
            {
                "token": tokenizer.decode([idx]),
                "token_id": int(idx),
                "logprob": float(lp) if lp != float("-inf") else None,
            }
            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist())
        ]
    }

    top_tokens_all_steps = topk_per_step(
        scores=outputs.scores,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        generated_ids=generated_ids,
        k=args.top_k,
    )

    metadata = {
        "prompt": args.prompt,
        "model_name": args.model,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "device": device,
        "timestamp": timestamp,
        "prompt_length": prompt_len,
        "generated_length": len(generated_ids) - prompt_len,
    }

    save_run_artifacts(
        run_dir=run_dir,
        generated_text=generated_text,
        prompt_hidden_states=prompt_hidden_states,
        prompt_attentions=prompt_attentions,
        prompt_logits=prompt_logits,
        prompt_hooks=prompt_hooks.to_dict(),
        fullseq_hooks=fullseq_hooks.to_dict(),
        gen_hidden_states=gen_hidden_states,
        gen_attentions=gen_attentions,
        gen_logits=gen_logits,
        top_tokens_first_step=top_tokens_first_step,
        top_tokens_all_steps=top_tokens_all_steps,
        metadata=metadata,
    )

    for h in hook_handles:
        h.remove()

    print("\n" + "=" * 60)
    print(f"PROMPT:\n  {args.prompt}")
    print("-" * 60)
    print(f"RESPONSE:\n  {generated_text[len(args.prompt):].strip()}")
    print("=" * 60 + "\n")
    print(f"Saved run to: {run_dir}")

    # Auto-run inspect_run.py to convert .pt files to JSON
    inspect_script = Path(__file__).parent / "inspect_run.py"
    if inspect_script.exists():
        print("\nRunning inspect_run.py...")
        result = subprocess.run(
            [sys.executable, str(inspect_script), "--run-dir", str(run_dir)],
            check=False,
        )
        if result.returncode != 0:
            print("Warning: inspect_run.py finished with errors.")
    else:
        print(f"Warning: inspect_run.py not found at {inspect_script}, skipping export.")

    # Auto-run build_token_matrix.py to stack all activations into one matrix
    matrix_script = Path(__file__).parent / "build_token_matrix.py"
    if matrix_script.exists():
        print("\nRunning build_token_matrix.py...")
        result = subprocess.run(
            [sys.executable, str(matrix_script), "--run-dir", str(run_dir)],
            check=False,
        )
        if result.returncode != 0:
            print("Warning: build_token_matrix.py finished with errors.")
    else:
        print(f"Warning: build_token_matrix.py not found at {matrix_script}, skipping.")


if __name__ == "__main__":
    main()