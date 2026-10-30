"""Stage FT-2 - single-speaker SFT of Qwen3-TTS on one 16 GB card.

Derived from the official ``finetuning/sft_12hz.py`` and ``dataset.py``
(Apache-2.0, (c) 2026 The Alibaba Qwen team). The batch layout in ``collate`` is the
official one, unchanged - it is what the talker's input format is. Everything else is
rewritten, for two reasons: the card, and four defects in the upstream training loop
that are silent (loss still falls) but audible.

WHAT IS FIXED RELATIVE TO THE OFFICIAL SCRIPT
---------------------------------------------
1. **Label alignment.** ``transformers``' ``ForCausalLMLoss`` already shifts labels by
   one. The official loop *also* shifts by hand (``inputs_embeds[:, :-1]`` against
   ``labels[:, 1:]``), so the model is trained to predict the codec frame two steps
   ahead. The sub-talker has the mirror problem: its 15 logits are matched against 14
   labels plus one ignore. Here the full unshifted tensors go in and the sub-talker
   loss is computed directly from the logits.  (upstream issue #278)
2. **Sub-talker conditioning.** The hidden state that *predicted* frame t sits at
   t-1, so the sub-talker must read ``hidden[codec_mask shifted by one]``, not
   ``hidden[codec_mask]``.  (upstream issue #337)
3. **The text projection.** Inference feeds text embeddings through
   ``talker.text_projection``; the official trainer adds the raw embedding straight to
   the codec embedding. On the 1.7B the shapes happen to match so nothing crashes -
   the model is simply trained on inputs it will never see again. (It is also why the
   0.6B, whose text width differs, crashes outright.)  (upstream issues #278, #336)
4. **The speaker encoder decays while you train.** Its output is detached, so it gets
   no gradient - but AdamW still applies weight decay to it every step, so the
   conditioning vector drifts away from the one written into the checkpoint. Here it
   is frozen, and the vector is extracted once from ref.wav and reused for every step,
   which is also exactly what CustomVoice inference will use.  (upstream issue #351)

Pass ``--label_alignment official`` to reproduce the upstream behaviour for an A/B.

WHAT IS DIFFERENT FOR THE HARDWARE
----------------------------------
* **sdpa, not flash-attention-2.** There is no flash-attn build for Blackwell on
  Windows; sdpa is the correct fallback and is what the eval script loads with too.
* **LoRA by default.** A full 1.7B fine-tune needs ~15 GB for weights + grads +
  moments before a single activation, and in pure bf16 an Adam step at these learning
  rates is smaller than the weight's own representable resolution, so most of it
  rounds away. An fp32 adapter over the frozen bf16 base fits in about 5 GB and
  actually applies its updates. ``--mode full`` is still here for the 0.6B (which
  fits in fp32) or for a bigger card.
* **Length-bucketed batches** and a **worst-case VRAM probe before step 0**: the
  longest batch in the dataset is run through a full forward/backward at startup, so
  an OOM happens in the first ten seconds rather than in epoch 3.
* **Checkpoints keep the official layout** (a full model directory with the speaker
  written into ``codec_embedding`` row 3000) so ``Qwen3TTSModel.from_pretrained`` and
  ``generate_custom_voice`` work unchanged - plus a small adapter file per epoch.

Everything is single-GPU on purpose: no accelerate, no deepspeed, one process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import ft_config as C

C.utf8_stdout()


# ---------------------------------------------------------------- data

class CodedDataset:
    """Rows from ``<split>_with_codes.jsonl``, tokenised once up front.

    The official dataset re-reads and re-mels ``ref.wav`` for every single item; with
    one shared reference that is ~10 000 redundant librosa calls per epoch. The
    reference is handled once, outside, and the speaker vector is passed in.
    """

    def __init__(self, rows, processor, config, max_frames: int | None = None):
        import torch

        self.torch = torch
        self.config = config
        self.items: list[dict] = []
        skipped = 0
        for row in rows:
            if "audio_codes" not in row:
                raise ValueError(f"row has no audio_codes: {row.get('audio', '<unknown>')}")
            codes = np.asarray(row["audio_codes"], dtype=np.int16)
            if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] == 0:
                raise ValueError(
                    f"audio_codes must have shape [frames, 16], got {codes.shape}: "
                    f"{row.get('audio', '<unknown>')}")
            frames = len(codes)
            if max_frames and frames > max_frames:
                skipped += 1
                continue
            text = f"<|im_start|>assistant\n{row['text']}<|im_end|>\n<|im_start|>assistant\n"
            ids = processor(text=text, return_tensors="pt", padding=True)["input_ids"]
            ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
            ids = ids[:, :-5]           # drop <|im_end|>\n<|im_start|>assistant\n
            self.items.append({
                "audio": row["audio"],
                "text_ids": ids.to(torch.long),
                # int16 is enough for a 2048-entry codebook and keeps 10k clips at ~12 MB
                "codes": codes,
            })
        self.skipped = skipped
        self.lengths = [8 + it["text_ids"].shape[1] + len(it["codes"]) for it in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "text_ids": it["text_ids"],
            "audio_codes": self.torch.from_numpy(it["codes"].astype(np.int64)),
        }

    def collate(self, batch):
        """The official collate_fn, minus the reference mels.

        Layout per row (t is the sequence axis):
            text channel   [ im_start assistant \\n ] [pad pad pad pad] [bos] text... [eos] pad...
            codec channel  [   0        0        0  ] [nothink think_bos think_eos SPK] [pad]
                           pad... [codec_bos] c0 c1 ... [codec_eos]
        Position 6 of the codec channel is masked out and overwritten with the speaker
        embedding by the caller.
        """
        torch = self.torch
        cfg = self.config
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        t = max(item_length) + 8
        b = len(batch)

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codecs = data["audio_codes"]
            audio_codec_0 = audio_codecs[:, 0]
            n_text = text_ids.shape[1]
            n_codec = audio_codec_0.shape[0]

            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = cfg.tts_pad_token_id
            input_ids[i, 7, 0] = cfg.tts_bos_token_id
            input_ids[i, 8:8 + n_text - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + n_text - 3, 0] = cfg.tts_eos_token_id
            input_ids[i, 8 + n_text - 2:8 + n_text + n_codec, 0] = cfg.tts_pad_token_id
            text_embedding_mask[i, :8 + n_text + n_codec] = True

            input_ids[i, 3:8, 1] = torch.tensor([
                cfg.talker_config.codec_nothink_id,
                cfg.talker_config.codec_think_bos_id,
                cfg.talker_config.codec_think_eos_id,
                0,                                    # speaker embedding slot
                cfg.talker_config.codec_pad_id,
            ])
            input_ids[i, 8:8 + n_text - 3, 1] = cfg.talker_config.codec_pad_id
            input_ids[i, 8 + n_text - 3, 1] = cfg.talker_config.codec_pad_id
            input_ids[i, 8 + n_text - 2, 1] = cfg.talker_config.codec_bos_id
            input_ids[i, 8 + n_text - 1:8 + n_text - 1 + n_codec, 1] = audio_codec_0
            input_ids[i, 8 + n_text - 1 + n_codec, 1] = cfg.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + n_text - 1:8 + n_text - 1 + n_codec] = audio_codec_0
            codec_0_labels[i, 8 + n_text - 1 + n_codec] = cfg.talker_config.codec_eos_token_id
            codec_ids[i, 8 + n_text - 1:8 + n_text - 1 + n_codec, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + n_text + n_codec] = True
            codec_embedding_mask[i, 6] = False
            codec_mask[i, 8 + n_text - 1:8 + n_text - 1 + n_codec] = True
            attention_mask[i, :8 + n_text + n_codec] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }


def length_batches(lengths: list[int], batch_size: int, seed: int, shuffle: bool,
                   bucket_mult: int = 64) -> list[list[int]]:
    """Batches of similar-length rows, in shuffled order.

    Sorting globally would make every batch identical every epoch; sorting inside
    large random windows keeps padding low without freezing the batch composition.
    """
    idx = list(range(len(lengths)))
    if not shuffle:
        idx.sort(key=lambda i: lengths[i])
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng = random.Random(seed)
    rng.shuffle(idx)
    window = batch_size * bucket_mult
    batches: list[list[int]] = []
    for start in range(0, len(idx), window):
        chunk = sorted(idx[start:start + window], key=lambda i: lengths[i])
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- model plumbing

@dataclass
class Plan:
    """What the run intends to do, printed before it does it."""
    model: str
    mode: str
    dtype: str
    params_total_b: float
    params_trainable_m: float
    weights_gib: float
    grads_gib: float
    optimizer_gib: float
    static_gib: float
    free_gib: float


def build_model(args, device):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16
    if args.mode == "full" and args.full_dtype == "fp32":
        dtype = torch.float32
    print(f"[model] loading {args.init_model_path}  dtype={dtype}  attn={args.attn}")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path, dtype=dtype, attn_implementation=args.attn)
    model = wrapper.model
    if model.speaker_encoder is not None:
        # The encoder is inference-only here (issue #351): its output is detached, and
        # leaving it in the optimizer lets weight decay shrink it step by step.
        model.speaker_encoder.requires_grad_(False)
        model.speaker_encoder.eval()
    return wrapper, model, dtype


def speaker_embedding(model, ref_path: Path, device, dtype, cache: Path | None):
    """One vector, from ref.wav, reused for every step and written to the checkpoint.

    Training conditioning and CustomVoice inference then use the same number, which is
    the whole point of insisting on a single ``ref_audio`` (guide s16).
    """
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    if cache is not None and cache.exists():
        emb = torch.load(cache, map_location="cpu")
        print(f"[speaker] reusing locked embedding from {cache.name}  |e|={emb.norm():.3f}")
        return emb.to(device).to(dtype)

    if model.speaker_encoder is None:
        raise SystemExit(
            "this checkpoint has no speaker_encoder (it is already a custom_voice "
            "model) and no cached embedding was found - point --init_model_path at a "
            "*-Base model, or resume in the run directory that holds spk_embedding.pt")

    import soundfile as sf

    wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != C.TARGET_SR:
        raise SystemExit(f"ref_audio must be {C.TARGET_SR} Hz, got {sr}")
    with torch.inference_mode():
        mels = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024,
                               num_mels=128, sampling_rate=24000, hop_size=256,
                               win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        emb = model.speaker_encoder(mels.to(device).to(dtype))
    emb = emb[:1].detach().clone()
    print(f"[speaker] locked embedding from {ref_path.name}  "
          f"{tuple(emb.shape)}  |e|={emb.float().norm():.3f}  ({wav.size / sr:.2f}s)")
    if cache is not None:
        torch.save(emb.cpu(), cache)
    return emb


def build_inputs(model, batch, spk, device, use_text_projection: bool):
    """Text + codec + speaker -> one embedding sequence, exactly as inference builds it."""
    talker = model.talker
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    codec_ids = batch["codec_ids"].to(device, non_blocking=True)
    text_mask = batch["text_embedding_mask"].to(device, non_blocking=True)
    codec_mask_e = batch["codec_embedding_mask"].to(device, non_blocking=True)
    codec_mask = batch["codec_mask"].to(device, non_blocking=True)

    text_emb = talker.model.text_embedding(input_ids[:, :, 0])
    if use_text_projection:
        # What generation does. Skipping it trains the talker on inputs it never sees.
        text_emb = talker.text_projection(text_emb)
    text_emb = text_emb * text_mask

    codec_emb = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_mask_e
    codec_emb[:, 6, :] = spk.to(codec_emb.dtype).expand(codec_emb.shape[0], -1)

    embeds = text_emb + codec_emb
    sub_emb = talker.code_predictor.get_input_embeddings()
    for i in range(1, 16):
        embeds = embeds + sub_emb[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)
    return embeds, codec_ids, codec_mask


def sub_talker_logits(talker, codec_ids, hidden):
    """``forward_sub_talker_finetune`` without its (misaligned) internal loss."""
    import torch

    groups = talker.config.num_code_groups
    parts = [hidden.unsqueeze(1), talker.get_input_embeddings()(codec_ids[:, :1])]
    sub_emb = talker.code_predictor.get_input_embeddings()
    for i in range(1, groups - 1):
        parts.append(sub_emb[i - 1](codec_ids[:, i:i + 1]))
    out = talker.code_predictor.forward_finetune(
        inputs_embeds=torch.cat(parts, dim=1), labels=None)
    return out.logits


def compute_loss(model, batch, spk, device, args, capture):
    """Total loss for one micro-batch, plus its two components and a token count."""
    import torch
    import torch.nn.functional as F

    embeds, codec_ids, codec_mask = build_inputs(
        model, batch, spk, device, args.text_projection)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["codec_0_labels"].to(device, non_blocking=True)
    talker = model.talker

    if args.label_alignment == "official":
        # Reproduce upstream: hand-shifted inputs *and* the internal shift.
        out = talker(inputs_embeds=embeds[:, :-1], attention_mask=attention_mask[:, :-1],
                     labels=labels[:, 1:], use_cache=False)
        hidden = capture["h"]
        sel_hidden = hidden[codec_mask[:, :-1]]
    else:
        out = talker(inputs_embeds=embeds, attention_mask=attention_mask,
                     labels=labels, use_cache=False)
        hidden = capture["h"]
        # The state that predicted frame t lives at t-1 (issue #337).
        sel_hidden = hidden[:, :-1][codec_mask[:, 1:]]

    sel_codes = codec_ids[codec_mask]
    logits = sub_talker_logits(talker, sel_codes, sel_hidden)
    vocab = talker.code_predictor.config.vocab_size
    if args.label_alignment == "official":
        # ForCausalLMLoss would pad-and-shift these; do the same to stay faithful.
        tgt = F.pad(sel_codes[:, 1:], (0, 1), value=-100)[:, 1:]
    else:
        tgt = sel_codes[:, 1:]
    sub_loss = F.cross_entropy(logits.float().reshape(-1, vocab), tgt.reshape(-1),
                               ignore_index=-100)
    loss = out.loss + args.sub_talker_weight * sub_loss
    n_frames = int(codec_mask.sum())
    return loss, out.loss.detach(), sub_loss.detach(), n_frames


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(out_dir: Path, base_dir: Path, state_dict, spk, speaker_name: str):
    """The official checkpoint layout: a full model dir this speaker can be called by."""
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy the tree but not the base weights - they are about to be overwritten.
    shutil.copytree(base_dir, out_dir, dirs_exist_ok=True,
                    ignore=lambda d, names: {"model.safetensors"} if Path(d) == Path(base_dir) else set())

    cfg = json.loads((Path(base_dir) / "config.json").read_text(encoding="utf-8"))
    cfg["tts_model_type"] = "custom_voice"
    talker_cfg = cfg.get("talker_config", {})
    talker_cfg["spk_id"] = {speaker_name: 3000}
    talker_cfg["spk_is_dialect"] = {speaker_name: False}
    cfg["talker_config"] = talker_cfg
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    sd = {k: v for k, v in state_dict.items() if not k.startswith("speaker_encoder")}
    key = "talker.model.codec_embedding.weight"
    weight = sd[key]
    weight[3000] = spk[0].detach().to(weight.device).to(weight.dtype)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(out_dir / "model.safetensors"))
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 2 ** 30
    print(f"[save] {out_dir}  ({size:.2f} GiB)")


def rotate(run_dir: Path, keep: int) -> None:
    ckpts = sorted(run_dir.glob("checkpoint-epoch-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[1]))
    for old in ckpts[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        print(f"[save] pruned {old.name} (--keep_last {keep})")


# ---------------------------------------------------------------- training

def evaluate(model, loader_batches, dataset, spk, device, args, capture) -> dict:
    import torch

    model.eval()
    totals = np.zeros(3)
    frames = 0
    with torch.no_grad():
        for indices in loader_batches:
            batch = dataset.collate([dataset[i] for i in indices])
            loss, main, sub, n = compute_loss(model, batch, spk, device, args, capture)
            totals += np.array([float(loss.detach()), float(main), float(sub)]) * n
            frames += n
    model.train()
    if not frames:
        return {}
    total, main, sub = totals / frames
    return {"val_loss": round(float(total), 4), "val_main": round(float(main), 4),
            "val_sub": round(float(sub), 4), "val_frames": frames}


def autotune_batch_size(model, dataset, spk, device, args, capture, candidates) -> int:
    """Run the *longest* rows in the dataset through fwd+bwd and keep what fits."""
    import torch

    order = sorted(range(len(dataset)), key=lambda i: -dataset.lengths[i])
    for size in candidates:
        if size > len(dataset):
            continue
        batch = dataset.collate([dataset[i] for i in order[:size]])
        try:
            torch.cuda.reset_peak_memory_stats()
            loss, *_ = compute_loss(model, batch, spk, device, args, capture)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            longest = max(dataset.lengths[i] for i in order[:size])
            print(f"[probe] batch {size:>3d} x {longest:>4d} tokens (worst case) "
                  f"-> peak {peak:.2f} GiB")
            return size
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[probe] batch {size} does not fit, trying smaller")
    raise SystemExit("even batch size 1 does not fit - use --mode lora, a smaller "
                     "--max_frames, or the 0.6B model")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Single-speaker Qwen3-TTS SFT, sized for a 16 GB card.")
    g = ap.add_argument_group("data")
    g.add_argument("--train_jsonl", default=str(C.coded_path("train")))
    g.add_argument("--val_jsonl", default=str(C.coded_path("val")))
    g.add_argument("--ref_audio", default=str(C.REF_OUT))
    g.add_argument("--max_frames", type=int, default=400,
                   help="drop clips longer than this many codec frames (~32 s)")
    g.add_argument("--max_val_batches", type=int, default=24)
    g.add_argument("--max_train_samples", type=int, default=None,
                   help="first N coded rows (smoke/debug runs only)")
    g.add_argument("--allow_small_dataset", action="store_true",
                   help="allow fewer than 100 training rows (smoke tests only)")

    g = ap.add_argument_group("model")
    g.add_argument("--init_model_path", default="1.7B",
                   help="1.7B | 0.6B | hub id | local checkpoint dir")
    g.add_argument("--speaker_name", default="kurisu")
    g.add_argument("--mode", choices=["lora", "full"], default="lora")
    g.add_argument("--full_dtype", choices=["bf16", "fp32"], default="bf16",
                   help="--mode full only; fp32 is numerically right but needs ~4x the room")
    g.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    g.add_argument("--lora_r", type=int, default=32)
    g.add_argument("--lora_alpha", type=float, default=64.0)
    g.add_argument("--lora_dropout", type=float, default=0.05)

    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=6)
    g.add_argument("--batch_size", default="auto", help="micro-batch, or 'auto' to probe")
    g.add_argument("--effective_batch", type=int, default=32,
                   help="clips per optimizer step; sets grad accumulation")
    g.add_argument("--lr", type=float, default=None,
                   help="default 1e-4 for lora, 1e-5 for full")
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g.add_argument("--warmup_ratio", type=float, default=0.03)
    g.add_argument("--weight_decay", type=float, default=None,
                   help="default 0.0 for lora, 0.01 for full")
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--sub_talker_weight", type=float, default=0.3)
    g.add_argument("--optimizer", choices=["auto", "adamw", "adamw8bit", "adafactor"],
                   default="auto")
    g.add_argument("--grad_checkpointing", choices=["auto", "on", "off"], default="auto")
    g.add_argument("--label_alignment", choices=["fixed", "official"], default="fixed")
    g.add_argument("--text_projection", type=lambda s: s.lower() != "off", default=True,
                   help="'off' reproduces the official (mismatched) training input")
    g.add_argument("--seed", type=int, default=20260821)

    g = ap.add_argument_group("run")
    g.add_argument("--out", default=None, help="run directory (default runs/<name>)")
    g.add_argument("--name", default=None)
    g.add_argument("--keep_last", type=int, default=3, help="checkpoint dirs to keep")
    g.add_argument("--eval_every", type=int, default=200, help="optimizer steps")
    g.add_argument("--log_every", type=int, default=10)
    g.add_argument("--vram_fraction", type=float, default=0.92)
    g.add_argument("--resume", default=None, help="path to state.pt, or 'auto'")
    g.add_argument("--tensorboard", action="store_true")
    g.add_argument("--force", action="store_true", help="ignore the VRAM plan warning")
    args = ap.parse_args()

    args.init_model_path = C.resolve_model(args.init_model_path)
    if args.lr is None:
        args.lr = 1e-4 if args.mode == "lora" else 1e-5
    if args.weight_decay is None:
        args.weight_decay = 0.0 if args.mode == "lora" else 0.01
    name = args.name or f"{args.speaker_name}-{args.mode}-r{args.lora_r}"
    run_dir = Path(args.out) if args.out else (C.RUNS / name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "state.pt").exists() and args.resume is None:
        raise SystemExit(
            f"{run_dir} already contains training state. Refusing to mix two runs; "
            "pass --resume auto to continue it, or choose a new --name/--out.")
    if (args.resume == "auto" and not (run_dir / "state.pt").exists()
            and (any(run_dir.glob("checkpoint-epoch-*")) or (run_dir / "train_log.jsonl").exists())):
        raise SystemExit(
            f"{run_dir} has old run artifacts but no resumable state.pt. Choose a new "
            "--name/--out instead of mixing experiments.")
    if args.resume and (run_dir / "state.pt").exists() and (run_dir / "args.json").exists():
        previous = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        immutable = ("init_model_path", "speaker_name", "mode", "lora_r", "lora_alpha",
                     "lora_dropout", "label_alignment", "text_projection", "optimizer")
        changed = [key for key in immutable if previous.get(key) != getattr(args, key)]
        if changed:
            details = ", ".join(
                f"{key}: {previous.get(key)!r} -> {getattr(args, key)!r}" for key in changed)
            raise SystemExit(
                f"resume settings do not match the saved run ({details}). Use a new "
                "--name/--out for a different model or adapter configuration.")

    device = C.vram_guard(args.vram_fraction)
    import torch
    from transformers import AutoConfig

    import ft_lora

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    wrapper, model, dtype = build_model(args, device)
    config = AutoConfig.from_pretrained(args.init_model_path)
    model.to(device)

    spk_cache = run_dir / "spk_embedding.pt"
    spk = speaker_embedding(model, Path(args.ref_audio), device, dtype, spk_cache)

    lora_info = None
    if args.mode == "lora":
        lora_info = ft_lora.inject(model, r=args.lora_r, alpha=args.lora_alpha,
                                   dropout=args.lora_dropout)
        ft_lora.freeze_base(model)
        model.to(device)
        print(f"[lora] r={args.lora_r} alpha={args.lora_alpha:g} on "
              f"{lora_info['modules_wrapped']} projections")
    else:
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(not n_.startswith("speaker_encoder"))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_param = 2 if dtype == torch.bfloat16 else 4

    # --- optimizer -------------------------------------------------------------
    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (decay if p_.ndim >= 2 else no_decay).append(p_)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]

    optim_name = args.optimizer
    if optim_name == "auto":
        # Full fine-tuning needs the small-state optimizers; LoRA is tiny either way.
        optim_name = "adamw" if args.mode == "lora" else "adamw8bit"
    if optim_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, lr=args.lr, betas=(0.9, 0.95))
            state_bytes = 2
        except Exception as exc:                       # noqa: BLE001
            print(f"[optim] bitsandbytes unavailable ({exc.__class__.__name__}); "
                  f"falling back to adafactor")
            optim_name = "adafactor"
    if optim_name == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(groups, lr=args.lr, scale_parameter=False,
                              relative_step=False, warmup_init=False)
        state_bytes = 0.05
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=False)
        state_bytes = 2 * (4 if dtype == torch.float32 or args.mode == "lora" else 2)

    plan = Plan(
        model=args.init_model_path, mode=args.mode, dtype=str(dtype).replace("torch.", ""),
        params_total_b=round(n_total / 1e9, 3), params_trainable_m=round(n_train / 1e6, 1),
        weights_gib=round(n_total * bytes_param / 2 ** 30, 2),
        grads_gib=round(n_train * (4 if args.mode == "lora" else bytes_param) / 2 ** 30, 2),
        optimizer_gib=round(n_train * state_bytes / 2 ** 30, 2),
        static_gib=0.0, free_gib=round(C.free_vram_gib(), 2))
    plan.static_gib = round(plan.weights_gib + plan.grads_gib + plan.optimizer_gib, 2)
    print(f"[plan] {plan.params_total_b:.3f}B params, {plan.params_trainable_m:.1f}M "
          f"trainable ({100 * n_train / n_total:.2f}%), optimizer={optim_name}")
    print(f"[plan] weights {plan.weights_gib:.2f} + grads {plan.grads_gib:.2f} + "
          f"optimizer {plan.optimizer_gib:.2f} = {plan.static_gib:.2f} GiB static, "
          f"{plan.free_gib:.2f} GiB free")
    if plan.static_gib > plan.free_gib * 0.8 and not args.force:
        raise SystemExit(
            f"the static footprint alone is {plan.static_gib:.1f} GiB of "
            f"{plan.free_gib:.1f} GiB free - activations still have to fit. Use "
            f"--mode lora, --optimizer adafactor, the 0.6B model, or --force.")

    grad_ckpt = args.grad_checkpointing == "on" or (
        args.grad_checkpointing == "auto" and (args.mode == "full" or n_total > 1e9))
    if grad_ckpt:
        model.talker.gradient_checkpointing_enable({"use_reentrant": False})
        print("[plan] gradient checkpointing on")

    # --- data ------------------------------------------------------------------
    processor = wrapper.processor
    t0 = time.time()
    train_rows = list(C.read_jsonl(Path(args.train_jsonl)))
    if args.max_train_samples is not None:
        train_rows = train_rows[:args.max_train_samples]
    if len(train_rows) < 100 and not args.allow_small_dataset:
        raise SystemExit(
            f"{args.train_jsonl} contains only {len(train_rows)} rows. This is usually "
            "a partial preparation file or the validation split. Finish ft01_prepare.py, "
            "or pass --allow_small_dataset for an intentional smoke test.")
    train_ds = CodedDataset(train_rows, processor, config, args.max_frames)
    val_ds = None
    if Path(args.val_jsonl).exists():
        val_ds = CodedDataset(list(C.read_jsonl(Path(args.val_jsonl))), processor,
                              config, args.max_frames)
    lens = np.array(train_ds.lengths)
    print(f"[data] {len(train_ds)} train clips ({train_ds.skipped} over --max_frames), "
          f"{len(val_ds) if val_ds else 0} val, tokenised in {time.time() - t0:.1f}s")
    print(f"[data] sequence length median {int(np.median(lens))}, "
          f"p95 {int(np.percentile(lens, 95))}, max {int(lens.max())}")

    capture: dict = {}

    def _hook(_module, _inputs, output):
        capture["h"] = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]

    model.talker.model.register_forward_hook(_hook)
    model.train()

    if args.batch_size == "auto":
        batch_size = autotune_batch_size(model, train_ds, spk, device, args, capture,
                                         [24, 16, 12, 8, 6, 4, 2, 1])
    else:
        batch_size = int(args.batch_size)
    accum = max(1, round(args.effective_batch / batch_size))
    batches = length_batches(train_ds.lengths, batch_size, args.seed, shuffle=True)
    # The final, partial accumulation window still performs an optimizer step.
    steps_per_epoch = max(1, math.ceil(len(batches) / accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    print(f"[plan] micro-batch {batch_size} x accum {accum} = {batch_size * accum} clips/step, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total, warmup {warmup}")

    val_batches = []
    if val_ds is not None:
        val_batches = length_batches(val_ds.lengths, batch_size, args.seed,
                                     shuffle=False)[:args.max_val_batches]

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)

    # --- resume ----------------------------------------------------------------
    start_epoch, global_step = 0, 0
    resume_path = None
    if args.resume == "auto":
        resume_path = run_dir / "state.pt" if (run_dir / "state.pt").exists() else None
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if args.mode == "lora":
            ft_lora.load_adapter(model, {k: v.to(device) for k, v in state["adapter"].items()})
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        random.setstate(state["py_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"[resume] {resume_path.name}: continuing from epoch {start_epoch}, "
              f"step {global_step}")

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))

    log_path = run_dir / "train_log.jsonl"
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    history: list[dict] = []
    base_dir = Path(args.init_model_path)
    if not base_dir.exists():          # a hub id - resolve to the snapshot on disk
        from huggingface_hub import snapshot_download
        base_dir = Path(snapshot_download(args.init_model_path))

    autocast = (torch.autocast("cuda", torch.bfloat16)
                if (args.mode == "full" and dtype == torch.float32)
                else torch.autocast("cuda", enabled=False))

    print(f"[run] {run_dir}")
    t_start = time.time()
    ema = None
    try:
        for epoch in range(start_epoch, args.epochs):
            batches = length_batches(train_ds.lengths, batch_size, args.seed + epoch,
                                     shuffle=True)
            t_epoch = time.time()
            running = np.zeros(3)
            running_frames = 0
            seen_frames = 0
            optimizer.zero_grad(set_to_none=True)
            for i, indices in enumerate(batches):
                batch = train_ds.collate([train_ds[j] for j in indices])
                with autocast:
                    loss, main, sub, n_frames = compute_loss(
                        model, batch, spk, device, args, capture)
                # Scale the final short accumulation window by its actual size. Without
                # this, the last update of every epoch is artificially weakened.
                window_start = (i // accum) * accum
                window_size = min(accum, len(batches) - window_start)
                (loss / window_size).backward()
                running += np.array([float(loss.detach()), float(main), float(sub)]) * n_frames
                running_frames += n_frames
                seen_frames += n_frames

                if (i + 1) % accum == 0 or i == len(batches) - 1:
                    lr = lr_at(global_step)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    loss_value = float(loss.detach())
                    ema = loss_value if ema is None else 0.98 * ema + 0.02 * loss_value
                    if global_step % args.log_every == 0:
                        done = i + 1
                        rate = done / (time.time() - t_epoch)
                        eta = (len(batches) - done) / max(rate, 1e-9)
                        peak = C.vram_used()[1]
                        print(f"  e{epoch} {done:>5d}/{len(batches)}  step {global_step:>5d}  "
                              f"loss {loss_value:.4f} (ema {ema:.4f})  "
                              f"main {float(main):.4f} sub {float(sub):.4f}  "
                              f"lr {lr:.2e}  {rate * batch_size:.1f} clip/s  "
                              f"vram {peak:.1f}G  eta {C.fmt_hms(eta)}", flush=True)
                        row = {"epoch": epoch, "step": global_step, "loss": loss_value,
                               "main": float(main), "sub": float(sub), "lr": lr,
                               "ema": ema, "vram_peak_gib": round(peak, 2)}
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                        if writer:
                            for k in ("loss", "main", "sub", "lr"):
                                writer.add_scalar(f"train/{k}", row[k], global_step)

                    if val_batches and args.eval_every and global_step % args.eval_every == 0:
                        stats = evaluate(model, val_batches, val_ds, spk, device, args, capture)
                        print(f"  [val] step {global_step}: {stats}", flush=True)
                        if writer:
                            writer.add_scalar("val/loss", stats["val_loss"], global_step)

            train_loss = running[0] / max(1, running_frames)
            stats = evaluate(model, val_batches, val_ds, spk, device, args, capture) \
                if val_batches else {}
            entry = {"epoch": epoch, "train_loss": round(float(train_loss), 4),
                     "steps": global_step, "seconds": round(time.time() - t_epoch, 1),
                     "frames": seen_frames, **stats}
            history.append(entry)
            print(f"[epoch {epoch}] {entry}", flush=True)

            ckpt = run_dir / f"checkpoint-epoch-{epoch}"
            need = 5.5 if "1.7B" in args.init_model_path else 3.0
            if C.free_disk_gib(run_dir) < need * 1.5:
                print(f"[save] only {C.free_disk_gib(run_dir):.1f} GiB free - "
                      f"skipping the full checkpoint, writing the adapter only")
            else:
                sd = (ft_lora.merged_state_dict(model) if args.mode == "lora"
                      else {k: v.detach().to("cpu") for k, v in model.state_dict().items()})
                save_checkpoint(ckpt, base_dir, sd, spk, args.speaker_name)
                del sd
                rotate(run_dir, args.keep_last)
            if args.mode == "lora":
                from safetensors.torch import save_file
                save_file(ft_lora.adapter_state_dict(model),
                          str(run_dir / f"adapter-epoch-{epoch}.safetensors"))
            state_path = run_dir / "state.pt"
            state_tmp = run_dir / "state.pt.tmp"
            torch.save({"epoch": epoch, "global_step": global_step,
                        "optimizer": optimizer.state_dict(),
                        "adapter": ft_lora.adapter_state_dict(model) if args.mode == "lora" else {},
                        "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()},
                       state_tmp)
            os.replace(state_tmp, state_path)
    except KeyboardInterrupt:
        print("\n[run] interrupted - state.pt holds the last completed epoch; "
              "restart with --resume auto")

    report = {
        "args": vars(args), "plan": asdict(plan), "lora": lora_info,
        "optimizer": optim_name, "batch_size": batch_size, "grad_accum": accum,
        "steps_per_epoch": steps_per_epoch, "grad_checkpointing": grad_ckpt,
        "train_clips": len(train_ds), "val_clips": len(val_ds) if val_ds else 0,
        "seq_len": {"median": int(np.median(lens)), "p95": int(np.percentile(lens, 95)),
                    "max": int(lens.max())},
        "history": history,
        "wall_seconds": round(time.time() - t_start, 1),
        "peak_vram_gib": round(C.vram_used()[1], 2),
        "run_dir": str(run_dir),
        "checkpoints": sorted(str(p) for p in run_dir.glob("checkpoint-epoch-*")),
        "next": "ft03_eval.py --ckpt all - loss alone does not tell you which epoch to "
                "ship (guide s33.2, upstream issue #179)",
    }
    C.write_report(f"ft_02_train_{name}.json", report)
    print(f"[run] done in {C.fmt_hms(time.time() - t_start)}, "
          f"peak VRAM {report['peak_vram_gib']:.1f} GiB")


if __name__ == "__main__":
    main()
