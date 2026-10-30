"""Check that the trainer's row layout matches what ``generate_custom_voice`` builds.

Writes a checkpoint the way ``train_sft.save_checkpoint`` does, from Base weights,
with three speakers and added tag tokens with a non-zero delta. Then it reloads that
checkpoint with the stock ``qwen_tts`` loader and, for every combination of text,
speaker and language, compares embeddings position by position:
- what ``generate_custom_voice`` hands to ``talker.generate``;
- what ``CodedDataset.collate`` + ``build_inputs`` build for training.

They must match up to and including ``tts_pad + codec_bos``. Run in fp32 on CPU; it
takes a few minutes and about 10 GB of RAM.

    python test_equivalence.py --base Qwen/Qwen3-TTS-12Hz-1.7B-Base --tmp /tmp/eqv
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft as T  # noqa: E402

TAGS = {"[sigh]": ["ため息", "sigh", "はぁ"], "[inhale]": ["息", "gasp"],
        "[laugh]": ["笑", "laugh"], "[nv]": ["声"], "[bg_noise]": ["雑音", "noise"]}
SPEAKERS = ["kurisu", "amadeus", "phone"]
TEXTS = ["こんにちは、世界。",
         "……[sigh]はぁ……そう。",
         "[bg_noise]岡部、聞こえる？",
         "それは[laugh]ふふっ、違うわ。[inhale]",
         "时间对于每个观测者来说都不一样。",
         "I am a scientist, not a magician."]
LANGS = ["Auto", "Japanese", "Chinese", "English"]


class Captured(Exception):
    pass


def captured_embeds(wrapper, text, speaker, language):
    talker = wrapper.model.talker
    original = talker.generate

    def fake(*_a, **kw):
        raise Captured(kw["inputs_embeds"].detach().clone())

    talker.generate = fake
    try:
        wrapper.generate_custom_voice(text=text, speaker=speaker, language=language,
                                      non_streaming_mode=True)
    except Captured as c:
        return c.args[0]
    finally:
        talker.generate = original
    raise RuntimeError("talker.generate was never reached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from tokenizers import AddedToken

    base_dir = Path(args.base) if Path(args.base).exists() else Path(snapshot_download(args.base))
    torch.manual_seed(0)
    wrapper = Qwen3TTSModel.from_pretrained(str(base_dir), dtype=torch.float32,
                                            attn_implementation="sdpa")
    model = wrapper.model
    tok = wrapper.processor.tokenizer
    first = len(tok)
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in TAGS])
    emb = model.talker.model.text_embedding.weight
    with torch.no_grad():
        for t, words in TAGS.items():
            src = [i for w in words for i in tok(w, add_special_tokens=False)["input_ids"]]
            emb[tok.convert_tokens_to_ids(t)] = emb[src].mean(0)
    new = T.NewTokens(first, len(TAGS), emb.shape[1], "cpu")
    with torch.no_grad():
        new.delta.normal_(0, 0.02)
    table = torch.randn(len(SPEAKERS), model.talker.config.hidden_size) * 0.5

    out = Path(args.tmp) / "ckpt"
    if out.exists():
        shutil.rmtree(out)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    T.save_checkpoint(out, base_dir, sd, table, SPEAKERS, new, tok)
    del wrapper, model, sd

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    tc = cfg["talker_config"]
    assert tc["spk_id"] == {"kurisu": 3000, "amadeus": 3001, "phone": 3002}, tc["spk_id"]
    assert set(tc["spk_is_dialect"]) == set(tc["spk_id"]), tc["spk_is_dialect"]

    w2 = Qwen3TTSModel.from_pretrained(str(out), dtype=torch.float32,
                                       attn_implementation="sdpa")
    m2 = w2.model
    m2.eval()
    tok2 = w2.processor.tokenizer
    failures = []

    # 1. tokenizer round trip: every tag is one id, whatever its neighbours
    for t in TAGS:
        want = tok2.convert_tokens_to_ids(t)
        for ctx in (f"{t}", f"、{t}", f"……{t}は", f"そう{t}。", f"[x]{t}"):
            ids = tok2(ctx, add_special_tokens=False)["input_ids"]
            if ids.count(want) != 1:
                failures.append(f"tokenizer: {t!r} in {ctx!r} -> {ids}")
    print(f"[tok] {len(tok2)} entries, tags at {[tok2.convert_tokens_to_ids(t) for t in TAGS]}")

    # 2. the folded delta landed in the saved embedding
    e2 = m2.talker.model.text_embedding.weight
    with torch.no_grad():
        for i, t in enumerate(TAGS):
            row = e2[tok2.convert_tokens_to_ids(t)]
            if float(row.norm()) < 1e-3:
                failures.append(f"embedding row for {t} is empty")

    # 3. speaker rows
    spk_table = m2.talker.model.codec_embedding.weight[3000:3000 + len(SPEAKERS)].detach()
    if not torch.allclose(spk_table, table.to(torch.bfloat16).float(), atol=1e-6):
        failures.append("saved speaker rows differ from the table")
    print(f"[spk] supported speakers: {m2.get_supported_speakers()}")

    # 4. layout equivalence
    config = m2.config
    spk_index = {n: i for i, n in enumerate(SPEAKERS)}
    checked = 0
    for text, speaker, lang in itertools.product(TEXTS, SPEAKERS, LANGS):
        ref = captured_embeds(w2, text, speaker, lang)
        row = {"audio": "x", "text": text, "audio_codes": [[1] * 16] * 3,
               "language": lang.lower(), "speaker": speaker}
        ds = T.CodedDataset([row], w2.processor, config, None, spk_index=spk_index)
        batch = ds.collate([ds[0]], train=False)
        with torch.no_grad():
            mine, _, _ = T.build_inputs(m2, batch, spk_table, "cpu", True, None)
        n = ref.shape[1]
        p = int(batch["spk_pos"][0]) + 2
        nb = ds.items[0]["text_ids"].shape[1] - 3
        if n != p + nb + 2:
            failures.append(f"length {text[:10]!r}/{speaker}/{lang}: generate {n}, trainer {p + nb + 2}")
            continue
        diff = (mine[:, :n] - ref).abs()
        if float(diff.max()) > 1e-5:
            bad = sorted(set(diff.max(-1).values[0].nonzero().flatten().tolist()))[:10]
            failures.append(f"embeds {text[:10]!r}/{speaker}/{lang}: max |d| "
                            f"{float(diff.max()):.2e} at positions {bad} (P={p}, n={n})")
        checked += 1
    print(f"[layout] {checked} combinations compared")

    # 5. Auto with p_auto=1 on an explicit-language row gives the Auto layout
    row = {"audio": "x", "text": TEXTS[0], "audio_codes": [[1] * 16] * 3,
           "language": "japanese", "speaker": "kurisu"}
    ds = T.CodedDataset([row], w2.processor, config, None, spk_index=spk_index, p_auto=1.0)
    if int(ds.collate([ds[0]], train=True)["spk_pos"][0]) != 6:
        failures.append("p_auto=1 did not switch the row to the Auto prefix")
    if int(ds.collate([ds[0]], train=False)["spk_pos"][0]) != 7:
        failures.append("eval collate did not keep the explicit language")

    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print("PASS: trainer rows == generate_custom_voice inputs for "
          f"{len(TEXTS)} texts x {len(SPEAKERS)} speakers x {len(LANGS)} languages")


if __name__ == "__main__":
    main()
