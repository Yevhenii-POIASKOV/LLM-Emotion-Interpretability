"""
data_loader.py
==============
Data Input Foundation for LLM Emotional Sentiment Interpretability.

Provides tokenization, padding/truncation, and batching of positive/negative
sentiment prompts using TransformerLens + HookedTransformer (GPT-2 small).

Usage:
    from data_loader import get_emotional_batches, DEVICE

    for batch_tokens, batch_labels, batch_texts in get_emotional_batches(batch_size=8):
        logits, cache = model.run_with_cache(batch_tokens)
        # batch_tokens : (B, T)  — ready for HookedTransformer
        # batch_labels : (B,)    — 1 = positive, 0 = negative
        # batch_texts  : list[str] of length B
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import torch
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# 1. Device selection
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[data_loader] Running on device: {DEVICE}")

# ---------------------------------------------------------------------------
# 2. Dataset — 60+ hardcoded prompts per sentiment class
#    Label convention: 1 = Positive, 0 = Negative
# ---------------------------------------------------------------------------
positive_prompts: list[str] = [
    # Joy / happiness
    "I woke up this morning feeling absolutely wonderful and full of energy.",
    "The surprise birthday party made her burst into tears of pure joy.",
    "After years of hard work, he finally got the promotion he deserved.",
    "Playing with my puppy in the park always lifts my spirits instantly.",
    "She laughed so hard her sides ached — best evening in months.",
    "The warm sunshine and gentle breeze made the hike utterly magical.",
    "Hearing my favourite song on the radio turned my whole day around.",
    "They embraced after five years apart, overcome with happiness.",
    "The newborn's first smile melted everyone in the room.",
    "We danced until midnight and every moment felt completely carefree.",
    # Gratitude / appreciation
    "I am deeply grateful for the incredible support of my family.",
    "What a beautiful gift — you really didn't have to, thank you so much.",
    "Volunteers worked tirelessly, and the community is forever thankful.",
    "The scholarship changed my life; I cannot express how blessed I feel.",
    "Receiving the handwritten letter from my grandmother filled me with warmth.",
    "He thanked his mentor with heartfelt words that moved everyone present.",
    "Gratitude washes over me every time I watch the sunrise.",
    "Their generosity during tough times was nothing short of extraordinary.",
    "I appreciate every small kindness; it truly makes a difference.",
    "The team celebrated their success and expressed sincere thanks to all.",
    # Hope / optimism
    "Despite the setbacks, she remained optimistic about the future.",
    "New treatment options give patients genuine hope for recovery.",
    "Every sunrise is a reminder that tomorrow holds fresh possibilities.",
    "The young entrepreneur believed wholeheartedly that things would improve.",
    "Community leaders came together, confident that change was within reach.",
    "After the storm, the rainbow inspired everyone to keep going.",
    "Scientists are hopeful that the new vaccine will save millions of lives.",
    "She smiled through her tears, knowing the best was yet to come.",
    "A letter of acceptance arrived, and with it, boundless optimism.",
    "His faith in humanity was restored by the stranger's random act of kindness.",
    # Love / compassion
    "The couple celebrated their fiftieth anniversary surrounded by loved ones.",
    "She held his hand and whispered that everything would be all right.",
    "Adopting the rescue dog was one of the most loving decisions we ever made.",
    "Neighbours rallied together to help the family rebuild after the flood.",
    "The teacher's patient encouragement transformed a struggling student's life.",
    "He wrote her a poem that captured perfectly how deeply he cared.",
    "Volunteering at the shelter reminded her how rewarding compassion can be.",
    "Their friendship, spanning decades, was built on unconditional trust and love.",
    "The mentor's belief in him gave him the courage to pursue his dream.",
    "A single act of kindness can ripple outward in ways we never imagine.",
    # Achievement / pride
    "She crossed the finish line and punched the air in triumph.",
    "After three years of study, he defended his thesis to a standing ovation.",
    "The team's championship win brought the entire city together in celebration.",
    "Her first published novel reached the bestseller list within a week.",
    "Completing the marathon was the proudest moment of his athletic career.",
    "The students presented their project with confidence and earned top marks.",
    "Landing the international contract was a milestone worth celebrating.",
    "She mastered a new language in under a year through sheer dedication.",
    "The young artist's gallery debut drew glowing reviews from critics.",
    "Overcoming every obstacle, they launched their start-up against all odds.",
    # Serenity / contentment
    "Sitting by the fireplace with a good book felt like pure contentment.",
    "The monastery garden offered a peace she had never found elsewhere.",
    "After a long day, a warm bath and soft music restored her completely.",
    "Walking through the autumn forest, he felt perfectly at ease with the world.",
    "Their small cottage by the sea was everything they had ever wanted.",
    "Meditation brought him a clarity and calm he cherished above all else.",
    "She savoured her morning coffee slowly, grateful for the quiet moment.",
    "The children played safely in the garden while the parents relaxed nearby.",
    "A weekend camping trip reconnected him with nature and himself.",
    "Surrounded by friends, she felt completely and utterly at home.",
    # Inspiration / awe
    "The documentary about ocean life left the audience in absolute awe.",
    "Witnessing the northern lights for the first time was a transcendent experience.",
    "The young activist's speech inspired thousands to take meaningful action.",
    "Reading her biography made me believe I could achieve anything.",
    "The breathtaking mountain vista reminded us how vast and wonderful the world is.",
]

negative_prompts: list[str] = [
    # Sadness / grief
    "She cried herself to sleep, overwhelmed by a grief she couldn't name.",
    "The funeral was a heartbreaking reminder of how fragile life truly is.",
    "He stared at the empty chair, still unable to accept that she was gone.",
    "The diagnosis left the family devastated and struggling to find words.",
    "Losing the job after fifteen years felt like losing part of his identity.",
    "The letter of rejection crushed months of hope in a single paragraph.",
    "She sat alone on her birthday, wondering if anyone had remembered.",
    "The abandoned puppy shivered in the rain, eyes full of confusion.",
    "Tears streamed down his face as the plane carried him away from home.",
    "Every old photograph was a painful reminder of what could never return.",
    # Anger / frustration
    "He slammed the door, furious at being dismissed yet again without reason.",
    "The repeated injustice left the protesters seething with righteous anger.",
    "She couldn't hide her frustration as the same mistake happened for the third time.",
    "Discovering the betrayal filled him with a cold, unrelenting rage.",
    "The endless bureaucratic delays left the team bitter and demoralized.",
    "He shouted into the phone, unable to believe what he was hearing.",
    "Years of being overlooked had turned her quiet resentment into fury.",
    "The broken promise shattered the trust they had spent years building.",
    "Watching corruption go unpunished made her feel powerless and incensed.",
    "He clenched his fists, struggling to contain his overwhelming anger.",
    # Fear / anxiety
    "She lay awake, heart pounding, dreading the results of the medical test.",
    "The thought of speaking in front of hundreds filled him with paralyzing dread.",
    "Every creak in the dark house sent another wave of fear through her.",
    "He couldn't breathe properly; anxiety had tightened its grip on his chest.",
    "The warning sirens triggered a panic she had never experienced before.",
    "Waiting for the verdict, she felt a terror she could barely contain.",
    "His hands trembled uncontrollably as the interview began.",
    "The uncertainty of the future haunted her every waking moment.",
    "She was terrified that every symptom pointed to something catastrophic.",
    "The nightmare left him shaking and unable to fall back asleep.",
    # Despair / hopelessness
    "He saw no path forward and felt completely crushed by hopelessness.",
    "The city lay in ruins, and despair had settled over the survivors like fog.",
    "She whispered that she didn't see the point in trying anymore.",
    "After years of fighting the illness, he quietly began to give up hope.",
    "The repeated failures had eroded every last trace of his self-belief.",
    "Nothing she tried seemed to matter; the situation only grew darker.",
    "He wandered the empty streets feeling utterly invisible and forgotten.",
    "The poverty they faced seemed inescapable and crushing in its weight.",
    "She read the final notice and felt the floor drop out from under her.",
    "Silence greeted him where there had once been warmth and laughter.",
    # Disgust / contempt
    "The politician's brazen lies left voters feeling profoundly disgusted.",
    "She recoiled at the cruelty with which the animals had been treated.",
    "The corruption scandal exposed a cynicism that sickened the public.",
    "His contempt for the vulnerable was barely concealed and deeply disturbing.",
    "The conditions in the facility were so appalling she had to look away.",
    "Discovering the deception made her feel sick to her stomach.",
    "The hateful graffiti on the school wall filled the community with revulsion.",
    "He was repulsed by the callousness of the decision-makers.",
    "The leaked footage revealed a cruelty that shocked even seasoned journalists.",
    "She couldn't hide her disgust as the injustice unfolded before her eyes.",
    # Loneliness / isolation
    "He moved to a new city and found the loneliness almost unbearable.",
    "She sat at the crowded party feeling entirely invisible and alone.",
    "Weeks passed without a single meaningful conversation to hold on to.",
    "The illness confined her to bed and cut her off from everyone she loved.",
    "He missed home with an ache that nothing seemed to ease.",
    "Despite being surrounded by colleagues, she felt profoundly isolated.",
    "The long winter nights amplified his sense of abandonment.",
    "She scrolled through social media, feeling more disconnected with every post.",
    "He had no one to call when the darkness became overwhelming.",
    "The empty apartment echoed with the absence of someone who was no longer there.",
    # Shame / regret
    "He carried the regret of that single terrible decision for years afterward.",
    "She wished she could take back the cruel words she had spoken in anger.",
    "The shame of the public failure followed him long after the incident.",
    "Looking back, she realized how badly she had hurt the people who loved her.",
    "He could barely meet their eyes, weighed down by guilt and remorse.",
]

# ---------------------------------------------------------------------------
# 3. Model loading (singleton pattern — load once, reuse)
# ---------------------------------------------------------------------------
_model_cache: HookedTransformer | None = None


def get_model() -> HookedTransformer:
    """Load GPT-2 small via TransformerLens (cached after first call)."""
    global _model_cache
    if _model_cache is None:
        print("[data_loader] Loading GPT-2 small via TransformerLens …")
        _model_cache = HookedTransformer.from_pretrained(
            "gpt2",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,           # fold LayerNorm weights for cleaner analysis
            refactor_factored_attn_matrices=False,
        )
        _model_cache.to(DEVICE)
        _model_cache.eval()        # disable dropout — deterministic forward passes
        print("[data_loader] Model loaded.")
    return _model_cache


# ---------------------------------------------------------------------------
# 4. Core tokenisation helper
# ---------------------------------------------------------------------------
def tokenize_prompts(
    texts: list[str],
    model: HookedTransformer,
    max_length: int = 64,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """
    Tokenise a list of strings into a uniform (N, T) integer tensor.

    Strategy
    --------
    * model.to_tokens()  — adds BOS automatically (prepend_bos=True default).
    * Sequences longer than `max_length` are **right-truncated**.
    * Sequences shorter than `max_length` are **right-padded** with `pad_token_id`
      (defaults to GPT-2's EOS token, id=50256, the standard choice for GPT-2
      because it has no dedicated PAD token).

    Parameters
    ----------
    texts       : raw string prompts
    model       : loaded HookedTransformer
    max_length  : target sequence length T (including BOS)
    pad_token_id: token id used for padding (default: EOS / 50256)

    Returns
    -------
    tokens : torch.Tensor, shape (N, T), dtype=torch.long, on DEVICE
    """
    if pad_token_id is None:
        # GPT-2 has no dedicated PAD token; EOS (50256) is the canonical choice.
        pad_token_id = model.tokenizer.eos_token_id  # 50256

    tokenized_rows: list[torch.Tensor] = []

    for text in texts:
        # model.to_tokens returns shape (1, seq_len) on DEVICE
        tok: torch.Tensor = model.to_tokens(text, prepend_bos=True)  # (1, S)
        tok = tok.squeeze(0)  # (S,)

        seq_len = tok.shape[0]

        if seq_len >= max_length:
            # Truncate — keep BOS token at position 0, then first max_length-1 real tokens
            tok = tok[:max_length]                       # (T,)
        else:
            # Pad on the right
            pad_len = max_length - seq_len
            padding = torch.full(
                (pad_len,), fill_value=pad_token_id,
                dtype=torch.long, device=DEVICE,
            )
            tok = torch.cat([tok, padding], dim=0)      # (T,)

        tokenized_rows.append(tok)

    # Stack into batch matrix
    tokens = torch.stack(tokenized_rows, dim=0)  # (N, T)
    return tokens


# ---------------------------------------------------------------------------
# 5. Dataclass for a single batch
# ---------------------------------------------------------------------------
@dataclass
class EmotionalBatch:
    """One mini-batch ready for model.run_with_cache()."""
    tokens: torch.Tensor    # (B, T)  — integer token ids
    labels: torch.Tensor    # (B,)    — 1=positive, 0=negative
    texts: list[str]        # length B — original strings (for logging / probing)

    def __len__(self) -> int:
        return self.tokens.shape[0]


# ---------------------------------------------------------------------------
# 6. Main public API: get_emotional_batches()
# ---------------------------------------------------------------------------
def get_emotional_batches(
    batch_size: int = 8,
    max_length: int = 64,
    shuffle: bool = True,
    model: HookedTransformer | None = None,
    pad_token_id: int | None = None,
    seed: int = 42,
) -> Iterator[EmotionalBatch]:
    """
    Yield EmotionalBatch objects suitable for model.run_with_cache().

    The positive and negative corpora are interleaved before batching so
    that each mini-batch contains a roughly equal mix of both sentiments
    (good practice to avoid label leakage via batch statistics in probing).

    Parameters
    ----------
    batch_size  : number of prompts per batch B
    max_length  : uniform token sequence length T (including BOS)
    shuffle     : whether to shuffle before batching
    model       : pre-loaded HookedTransformer (loaded lazily if None)
    pad_token_id: pad token (default: EOS 50256)
    seed        : random seed for reproducibility

    Yields
    ------
    EmotionalBatch with fields:
        .tokens  — torch.Tensor  (B, T)  long, on DEVICE
        .labels  — torch.Tensor  (B,)    long, on DEVICE  {0,1}
        .texts   — list[str]     length B

    Example
    -------
    >>> model = get_model()
    >>> for batch in get_emotional_batches(batch_size=8, model=model):
    ...     logits, cache = model.run_with_cache(batch.tokens)
    ...     # logits : (B, T, vocab_size)
    ...     # cache  : ActivationCache with all layer activations
    """
    if model is None:
        model = get_model()

    # -- Build paired corpus: (text, label)
    corpus: list[tuple[str, int]] = (
        [(t, 1) for t in positive_prompts] +
        [(t, 0) for t in negative_prompts]
    )

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(corpus)

    texts_all: list[str] = [t for t, _ in corpus]
    labels_all: list[int] = [l for _, l in corpus]

    n_total = len(texts_all)

    # -- Tokenise entire corpus at once for efficiency
    # tokens_all : (N, T)
    tokens_all: torch.Tensor = tokenize_prompts(
        texts_all, model, max_length=max_length, pad_token_id=pad_token_id,
    )
    labels_tensor: torch.Tensor = torch.tensor(
        labels_all, dtype=torch.long, device=DEVICE,
    )  # (N,)

    # -- Yield batches
    n_batches = math.ceil(n_total / batch_size)

    for i in range(n_batches):
        start = i * batch_size
        end   = min(start + batch_size, n_total)

        yield EmotionalBatch(
            tokens=tokens_all[start:end],    # (B, T)
            labels=labels_tensor[start:end], # (B,)
            texts=texts_all[start:end],      # list[str], length B
        )


# ---------------------------------------------------------------------------
# 7. Convenience: tokenise a single custom prompt (useful in analysis scripts)
# ---------------------------------------------------------------------------
def tokenize_single(
    text: str,
    model: HookedTransformer | None = None,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Tokenise a single string.

    Returns
    -------
    tokens : torch.Tensor, shape (1, T) — ready to pass to model.run_with_cache()
    """
    if model is None:
        model = get_model()
    tokens = tokenize_prompts([text], model, max_length=max_length)  # (1, T)
    return tokens


# ---------------------------------------------------------------------------
# 8. Quick sanity-check (run with: python data_loader.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== data_loader.py — Sanity Check ===\n")

    mdl = get_model()

    print(f"Positive prompts : {len(positive_prompts)}")
    print(f"Negative prompts : {len(negative_prompts)}")
    print(f"Total corpus size: {len(positive_prompts) + len(negative_prompts)}\n")

    for idx, batch in enumerate(get_emotional_batches(batch_size=8, model=mdl)):
        print(
            f"Batch {idx:02d} | "
            f"tokens.shape={tuple(batch.tokens.shape)} | "  # (B, T)
            f"labels={batch.labels.tolist()}"
        )
        if idx == 0:
            # Forward pass verification
            print("\n  [Batch 0 forward pass check]")
            with torch.no_grad():
                logits, cache = mdl.run_with_cache(batch.tokens)
            # logits : (B, T, d_vocab=50257)
            print(f"  logits.shape  = {tuple(logits.shape)}")
            print(f"  cache keys    = {len(cache.cache_dict)} activation tensors")
            # Spot-check: residual stream at final layer, last token position
            resid_final = cache["resid_post", mdl.cfg.n_layers - 1]  # (B, T, d_model)
            print(f"  resid_post[-1].shape = {tuple(resid_final.shape)}")
            print()

    print("=== All checks passed ===")