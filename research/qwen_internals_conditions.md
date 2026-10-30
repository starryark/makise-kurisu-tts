# Qwen3-TTS internals: per-row conditions for the Kurisu fine-tune

Date: 2026-10-28. Scope: how to add per-row **language id**, **instruct text**, **multiple speakers** and **non-verbal tags** to `kurisu_v2/pod/train_sft.py`, at the code level. This is a plan only. No project code was changed.

## Sources, and how they were read

- **`qwen-tts==0.1.1` wheel** (PyPI upload 2026-02-06). Unzipped to a scratch directory and read only; nothing from it was imported or executed. Line numbers below refer to this wheel:
  - `qwen_tts/inference/qwen3_tts_model.py` (cited as **QM**)
  - `qwen_tts/core/models/modeling_qwen3_tts.py` (**MD**)
  - `qwen_tts/core/models/configuration_qwen3_tts.py` (**CF**)
- **GitHub `QwenLM/Qwen3-TTS@main`** (fetched raw, plus `gh api` for commits, issues and PRs).
  - The two model files above are **byte-identical** on `main` and in the 0.1.1 wheel (`diff` is empty).
  - `pyproject.toml` on main still says `0.1.1`.
  - Fine-tuning files are cited as **UP-sft** (`finetuning/sft_12hz.py`) and **UP-ds** (`finetuning/dataset.py`).
- **HF `config.json`** for 1.7B-Base, 1.7B-CustomVoice, 1.7B-VoiceDesign, 0.6B-Base and 0.6B-CustomVoice. **Tokenizer data** (`vocab.json`, `merges.txt`, `tokenizer_config.json`) from 1.7B-Base; CustomVoice's `tokenizer_config.json` is identical.
- **Weight evidence.** Only the safetensors headers and a few tensors (codec embedding, slices of the text embedding, six projection matrices) were range-fetched from Base, CustomVoice, VoiceDesign and our published `starrydark/Kurisu_Qwen3_TTS`. They were parsed as raw bf16 with numpy.
- **Tokenization.** I rebuilt the Qwen2 byte-level BPE with the locally installed `tokenizers` 0.22.2 from the vocab and merges data (Qwen2 pre-tokenizer regex, NFC normalizer, the 33 added tokens). All 33 added-token ids round-trip.
- **Tech report.** arXiv 2601.15621, HTML version (§3.1-3.3).
- **Our code** is cited as **TS** (`kurisu_v2/pod/train_sft.py`), **APP** (`Kurisu_Qwen3_TTS/app.py`) and **P03** (`kurisu_v2/pod/p03_build.py`).

---

## 0. Key facts

| Question | Answer |
|---|---|
| Prefix, `language="Auto"` | `[role×3][nothink think_bos think_eos SPK pad][body…][tts_eos][codec_bos]…`. The speaker slot is at index 6 and the text body starts at 8. **This is exactly what TS trains today.** |
| Prefix, explicit language | `[role×3][think think_bos LANG think_eos SPK pad][body…]…`. The speaker slot is at 7 and the body starts at 9. |
| Instruct | Tokenized `<|im_start|>user\n{instruct}<|im_end|>\n` (not trimmed), text channel only, **prepended before the role tokens**, so every index shifts by `n_ins`. |
| Does `generate_custom_voice` pass instruct to our checkpoint? | **Yes.** Only 0.6B models drop it (`"1b7" in "0b6"` is False). Nothing checks that the model was fine-tuned from Base. Today the model has never seen an instruct, so the behaviour is untrained. |
| Base and instruct | The Base wrapper has no instruct argument. The Qwen collaborator in issue #25: the 12Hz Base's "control capability is relatively unstable", so it is **not supported**. In other words it is latent, not absent. |
| Can about 1,000 rows teach instruct from Base? | For a single global, acoustically salient flag (device EQ), probably, but not at our current regime. Our v2 moved weights **0.3-0.6 %** from Base; CustomVoice moved **3-7 %** and VoiceDesign **8-21 %**. Run the zero-cost probe (§3.5) first. |
| `[sigh]` and similar tags in the text vocab | 3-5 BPE pieces, and **the split depends on the neighbouring characters**: `[s` `igh` `]` on its own, but `[` `s` `igh` `]` after `、` or `…`. `[laugh]` becomes `[` `la` `ugh` `]`. No tag is a single token. |
| Spare text-token rows | `text_vocab_size` is 151936 and the last defined token is 151675, so **ids 151676-151935 (260 rows) exist with no token**. They look randomly initialised (norm ≈ 0.31, pairwise cosine 0). Newly added tokens land there, so **no embedding resize is needed**. |
| Spare codec rows | Of rows 2048-3071, **716 have never been trained** (norm < 0.05 in both Base and CustomVoice), including 2990-3009 and 3011-3060. Rows 3000-3004 are safe for new speakers. The codec ids generate() suppresses at sampling time are 2048-3071, so these rows can only ever be inputs. |
| Upstream after 0.1.1 | One commit (`022e286`, a sub-talker mask change in `sft_12hz.py`). Language id, instruct and multi-speaker support exist only in **open, unmerged PRs**: #361 (language, with a layout bug), #384 (multi-speaker) and #281 (voice-clone instruct). |

---

## 1. Language ids in the talker prefix

### 1.1 Mapping (1.7B config values; the CF defaults are different, so ignore them)

The CF defaults (`codec_think_id=4202`, `codec_pad_id=4196`, … at CF:393-399) belong to a different vocabulary. **Always read the values from `config.json`.** The advice in issue #323, which uses id 4206, is out of range for this vocabulary.

| Token | Codec id | Notes |
|---|---|---|
| `codec_pad_id` | 2148 | |
| `codec_bos_id` | 2149 | |
| `codec_eos_token_id` | 2150 | The only id in 2048-3071 that sampling may emit (MD:2059-2063) |
| `codec_think_id` | 2154 | |
| `codec_nothink_id` | 2155 | |
| `codec_think_bos_id` | 2156 | |
| `codec_think_eos_id` | 2157 | |

`codec_language_id` (identical in Base and VoiceDesign; CustomVoice adds the two dialect entries):

| Language | Id | Language | Id |
|---|---|---|---|
| english | 2050 | korean | 2064 |
| german | 2053 | russian | 2069 |
| spanish | 2054 | italian | 2070 |
| chinese | 2055 | portuguese | 2071 |
| japanese | 2058 | sichuan_dialect *(CV only)* | 2062 |
| french | 2061 | beijing_dialect *(CV only)* | 2074 |

Weight notes:
- In Base, every listed language row is trained (norm 1.05-1.27), and so is `sichuan_dialect` (1.20). Row 2074 (`beijing_dialect`) is **not** trained in Base (norm 0.022).
- Many **unlisted** rows near the language block are also trained: 2049, 2051, 2052, 2056, 2057, 2059, 2060, 2063, 2065-2068, 2072, 2073, 2075-2147 (partly), and 150 rows in 2248-2485 (norm about 0.3).
  - Their meaning is unknown (likely other languages or dialects, or pre-training control tags). **Do not repurpose them.**

### 1.2 Supported list and the dialect override

- **Supported languages** are `["auto"]` plus every `codec_language_id` key that does not contain `"dialect"` (MD:1830-1834). The wrapper validates against that list, case-insensitively (QM:141-163, 815).
  - Our checkpoint copies Base's config, so it supports auto plus the 10 languages.
- **Language id** (MD:2110-2116): `"auto"` gives `language_id = None`; any other value gives `codec_language_id[lang.lower()]`.
- **Dialect override** (MD:2118-2122):
  - It applies when `language.lower() in ["chinese", "auto"]`, a speaker is given, and `spk_is_dialect[speaker]` is not `False`.
  - Then `language_id = codec_language_id[spk_is_dialect[speaker]]`.
  - So for a dialect speaker, Auto becomes think mode with the dialect id.
- **Consequences for us:**
  - **Every key in `spk_id` must also be in `spk_is_dialect`.** Otherwise MD:2120 raises `KeyError` whenever language is Chinese or Auto.
  - The override only fires for Chinese and Auto, so it **cannot carry a channel condition (Amadeus or phone) on Japanese rows**. That rules it out for (c).
  - A value such as `"chinese"` would technically work for a `zh_dub` speaker (Auto would then become think+zh). It also forces the zh id on Auto requests with Japanese text, so leave every value `False`.

### 1.3 Exact prefix layouts

The relevant code:
- MD:2135-2147 builds `codec_prefill_list`.
- MD:2149-2172 appends `[SPK]` if a speaker exists, then `[pad, bos]`.
- MD:2177-2186: role = `text_proj(input_id[:, :3])` (`<|im_start|>`, `assistant`, `\n`; no codec embedding added). Then a text track of `tts_pad × (len-2) + tts_bos` is added to `codec[:, :-1]`.
- MD:2199-2227: in non-streaming mode the body is `text_proj(input_id[:, 3:-5]) + codec_pad`, then `tts_eos + codec_pad`, then `tts_pad + codec_bos`. Generation starts after that last position.
- During decoding, every step's input is the sum of all 16 codebook embeddings plus `tts_pad` (MD:1687-1692).

**`generate_custom_voice`, non-streaming (which is what APP uses), `language="Auto"`:**

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 … 8+n-1 | 8+n | 8+n+1 | 8+n+2 … |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text | `<\|im_start\|>` | `assistant` | `\n` | tts_pad | tts_pad | tts_pad | tts_pad | **tts_bos** | body tokens | **tts_eos** | tts_pad | tts_pad |
| codec | – | – | – | **nothink** | think_bos | think_eos | **SPK** | pad | pad | pad | **codec_bos** | c0+…+c15 → codec_eos |

**Explicit language** (for example `"Japanese"` → 2058): the codec slots at positions 3-8 become `think think_bos LANG think_eos SPK pad`. The text track over those positions is `tts_pad×5, tts_bos`, the speaker slot is at **7**, and the body starts at **9**.

**General formula.** It covers every case below; `n_ins = 0` when there is no instruct.

```
[instruct n_ins (text only)] [role 3 (text only)] [prefix k] [body n] [tts_eos] [codec_bos] [codes] [codec_eos]
k = 5 (Auto: nothink think_bos think_eos SPK pad)   or   6 (lang: think think_bos LANG think_eos SPK pad)
text over prefix = tts_pad×(k-1), tts_bos      spk_pos = n_ins + 3 + (k-2)      body_start P = n_ins + 3 + k
```

**Other entry points:**

| Entry point | Speaker slot | Prefix |
|---|---|---|
| `generate_voice_design` | None, so no SPK slot (MD:2166-2168) | Auto: `nothink think_bos think_eos pad`, body at 7; explicit language: body at 8 |
| `generate_voice_clone`, x-vector mode | Speaker is None, so the dialect check is skipped; SPK is the ref x-vector (MD:2103-2104) | Same codec layout as custom_voice |
| `generate_voice_clone`, ICL | As above | After `role+prefix` comes `[ref_text + text + tts_eos] + codec_pad`, then `[codec_bos + ref codes] + tts_pad` (MD:1968-2019, 2188-2197) |

`generate_voice_clone` defaults to `non_streaming_mode=False` (QM:478); custom_voice and voice_design default to `True` (QM:642, 738).

In streaming mode the first body token rides on the `codec_bos` position and the rest arrive as `trailing_text_hidden` during decoding (MD:2228-2232, 1689-1692). That is a different layout from training, so **keep `non_streaming_mode=True`** in the Space.

---

## 2. Instruct

### 2.1 Format and injection

- The text is built by `_build_instruct_text(ins) = f"<|im_start|>user\n{ins}<|im_end|>\n"` (QM:275-276) and tokenized whole, with no `[:-5]` trim (QM:820-825 for custom_voice, QM:710-715 for voice_design). An empty string or `None` means no instruct.
- `generate()` builds `text_projection(text_embedding(instruct_ids))` and appends it **first** to the row's embedding list (MD:2076-2080). The main prefix is appended after it (MD:2233), and the two are concatenated (MD:2236-2237).
- **No codec embedding is added at the instruct positions.** `generate_custom_voice` and `generate_voice_design` inject it identically.
- **Example.** `instruct="Speak like a digital recording played through a phone."` tokenizes to 15 ids: `[151644, 872('user'), 198, …, 13('.'), 151645, 198]`.

### 2.2 Which checkpoints honour instruct

| Checkpoint | Wrapper behaviour | Officially trained? |
|---|---|---|
| 1.7B-CustomVoice | `generate_custom_voice` passes it (QM:801, 829-836) | Yes. README table: "Instruction Control ✅"; the paper's InstructTTSEval "target speaker" results |
| 0.6B-CustomVoice | QM:799-800: `if tts_model_size in "0b6": instruct = None` (a substring test, effectively `== "0b6"`) | No (README: no ✅) |
| 1.7B-VoiceDesign | `generate_voice_design` always passes it (QM:686, 719-725) | Yes |
| Base (0.6B and 1.7B) | `generate_voice_clone` has **no instruct parameter**, and the call passes no `instruct_ids` (QM:470-480, 603-610). Lower-level `model.generate(instruct_ids=…)` would still accept it. Open PR #281 pre-wires it for a future 25Hz model. | "Not currently support[ed]", because 12Hz Base control is "relatively unstable" (collaborator, issue #25, 2026-01-26) |
| **Our checkpoint** (custom_voice, `tts_model_size: "1b7"`, fine-tuned from Base) | `generate_custom_voice` **passes it through** to `model.generate`; nothing checks the origin | Never trained with instruct |

### 2.3 Was Base trained with instruct?

The tech report does not split the data by checkpoint, but:
- §3.2: "All data is formatted in ChatML to standardize inputs and support controllable speech generation" (pre-training S1-S3).
- §3.3: "we prepend user-provided instructions containing fine-grained control signals to the input sequences".
- Post-training is DPO, then GSPO, then "lightweight speaker fine-tuning on the base model … further improving … controllability".

Together with issue #25 ("unstable", not "absent") and issue #14 (a 25Hz *VoiceEditing* model is promised for clone + instruct), the reading is: **instruction-formatted data was in pre-training, so the 12Hz Base has a weak, unreliable instruct prior.** It is not released as an instruct model.

Issue #248 adds that CustomVoice's instruct covers emotion and prosody, but not dialect switching.

### 2.4 Weight evidence, and an estimate for about 1,000 conditioned rows

Relative change ‖ΔW‖/‖W‖ against 1.7B-Base, measured on the same six tensors:

| Tensor | CustomVoice | VoiceDesign | **our v2** (lr 5e-6, 240 steps) |
|---|---|---|---|
| layers.0 q_proj | 0.063 | 0.213 | **0.0040** |
| layers.27 q_proj | 0.067 | 0.121 | **0.0036** |
| layers.27 v_proj | 0.062 | 0.091 | **0.0041** |
| text_projection fc1 | 0.042 | 0.106 | **0.0026** |
| text_projection fc2 | 0.026 | 0.075 | **0.0027** |
| codec_head | 0.069 | 0.084 | **0.0058** |

Other observations:
- **Text embedding rows 0-1999 are bit-identical** across Base, CV and VD (0 differing bf16 elements). In the tail, only `tts_pad/bos/eos` (151671-151673) moved, by about 1e-3.
  - The weights cannot tell us why. Possible causes: the embedding was frozen and the three TTS rows were handled separately, or the updates were below bf16 resolution.
- Either way, instruct understanding in CV/VD comes from Base's text pathway plus **3-21 % drift of the talker**, not from new text embeddings.
- Our v2 is about **10-20× lighter** than the CustomVoice branch.
- In our v2, the codec rows `think`, `ja`, `en` and `zh` show **exactly zero** drift: our training has never used them. `nothink` (0.6 %), `pad` and `bos` moved.

**Estimate.** The Amadeus and phone conditions are a single global attribute, present on every frame and acoustically obvious (spectral tilt and band-limiting). About 1,000 rows of roughly 60 frames each give about 60k supervised frames under the condition.
- Learning a discriminative prefix → global-colour mapping from that is plausible for a 1.7B LM that already has a latent instruct prior.
- At **lr 5e-6 × 1 epoch** (0.3-0.6 % drift), I expect it to be **weak or leaky**: the default voice picks up some EQ, and the instruct effect is partial.
- Expect to need either a higher LR on the attention projections (or LoRA on top at 1e-4) or 2-3 epochs over the conditioned subset. Verify with the tilt metric (§5.6).
- **Design choice.** If instruct appears **only** on Amadeus rows, the model learns "any instruct → device voice". That is fine for a flag, but it overwrites whatever emotion-instruct prior Base has. Varied instruct text on ordinary rows would be needed to keep both.

### 2.5 Step 0: a zero-cost probe on the published checkpoint

The current `starrydark/Kurisu_Qwen3_TTS` already accepts `instruct` (1b7, custom_voice). Run:
- `generate_custom_voice(text, speaker="kurisu", language="Auto", instruct=X)` for X in {none, "用电话听筒里的声音说", "Speak as if heard through a telephone.", "用激动的语气说"}
- 20 texts × 3 seeds for each X.

Measure spectral tilt and bandwidth (the P03 tilt metric) and CER. Any systematic shift means the latent prior survived and instruct training starts warm. No shift (or CER damage) means instruct must be learned almost from scratch, so prefer the speaker-row mechanism for (c).

---

## 3. Tokenization

### 3.1 Tags and emoji in the Qwen text BPE

Reproduced from `vocab.json`, `merges.txt` and the Qwen2 pre-tokenizer regex. `Ġ` marks a leading space.

| String | ids | pieces |
|---|---|---|
| `[sigh]` | 11880, 1090, 60 | `[s` `igh` `]` |
| `␠[sigh]` | 508, 82, 1090, 60 | `␠[` `s` `igh` `]` |
| `これが、[sigh]運命…` | …5373, **58, 82, 1090, 60**, 101452… | `、` `[` `s` `igh` `]` `運` |
| `…[sigh]もういいわ。` | 1940, **58, 82, 1090, 60**, 128778… | `…` `[` `s` `igh` `]` `もう` |
| `[gasp]` / `これが[gasp]、` | 32512, 13367, 60 | `[g` `asp` `]` |
| `[breath]` | 18483, 265, 587, 60 | `[b` `re` `ath` `]` |
| `[pause]` | 11407, 3454, 60 | `[p` `ause` `]` |
| `[laugh]` | 58, 4260, 7443, 60 | `[` `la` `ugh` `]` |
| `␠[laugh]␠` (after `。`) | 508, 4260, 7443, 60, then `␠当` 84897 | `␠[` `la` `ugh` `]` … |
| `(sigh)` | 1141, 1090, 8 | `(s` `igh` `)` |
| `<sigh>` | 44047, 1090, 29 | `<s` `igh` `>` |
| `<\|sigh\|>` | 27, 91, 82, 1090, 91, 29 | six pieces (it is not a special token) |
| `{sigh}` / `*sigh*` | 3 pieces each | `{s` `igh` `}` / `*s` `igh` `*` |
| `（ため息）` | 9909, 124616, 22226, 7552 | `（` `ため` `息` `）` (the model will try to *read* ため息) |
| `【ため息】` / `［ため息］` | 4 pieces each | |
| `……` | 14053 | **one token** (the script's native pause) |
| `😂` / `😔` | 144185 / 145379 | one token each |
| `😮‍💨` | 145596, 378, 235, 145711 | `😮` + ZWJ as 2 byte pieces + `💨` |

Why the split varies:
- The pre-tokenizer regex (`[^\r\n\p{L}\p{N}]?\p{L}+ | ?[^\s\p{L}\p{N}]+…`) glues `[` to a preceding punctuation run (`、[`, `…[`), or to the following letters when the tag starts a chunk (`[sigh`).
- So the same tag gets different ids depending on its neighbours, and its pieces (`igh`, `ause`, `la`, `ugh`) are shared with ordinary words.

That is the strongest argument against plain-text tags. It is also consistent with v1's runaway generations clustering on tagged prompts (P03:31-37).

### 3.2 Reserved and unused text tokens

- `vocab.json` holds 151643 regular tokens (0-151642). `tokenizer_config.json` defines 33 added tokens, 151643-151675:
  - `<|endoftext|> <|im_start|> <|im_end|>`
  - `<|object_ref_start/end|> <|box_start/end|> <|quad_start/end|> <|vision_start/end/pad|> <|image_pad|> <|video_pad|>`
  - `<tool_call> </tool_call> <|fim_*|>×4 <|repo_name|> <|file_sep|> <tool_response> </tool_response> <think> </think>`
  - `<|audio_start|> <|audio_end|>`
  - `<tts_pad>`=151671, `<tts_text_bos>`=151672, `<tts_text_eod>`=151673, `<tts_text_bos_single>`=151674, `<|audio_pad|>`=151675
- `<tts_text_bos_single>` and `<|audio_pad|>` are **never referenced** by the `qwen_tts` code.
- There are **no** `<|extra_N|>` tokens.
- The `text_embedding` has **151936 rows** (`text_vocab_size`), so rows **151676-151935 (260) exist with no token**.
  - Measured in Base, CV and VD (identical): norm 0.298-0.321 (regular rows about 0.61), mean pairwise cosine 0.000. They are untrained random rows inherited from the Qwen3 LM.
  - The added-token rows 151643-151675 are trained (norm 0.13-0.70).
- **Implication.** `tokenizer.add_tokens(...)` assigns ids from `len(tokenizer) = 151676` upward, inside the existing matrix. **There is no `resize_token_embeddings` and no change to any tensor shape.**
- Repurposing an existing special (`<|box_start|>`, `<|quad_start|>`, …) is also atomic and needs no tokenizer change, but those rows carry vision/grounding priors from the LM.

### 3.3 Spare codec-embedding rows

- `codec_embedding` is `[3072, 2048]`: ids 0-2047 are codebook 0 and 2048-3071 are control rows.
- Sampling suppresses 2048-3071 except `codec_eos` (MD:2059-2063), so control rows are **input-only**. A new row can never be emitted.
- **716 control rows were never trained** (norm < 0.05 in both Base and CustomVoice).
  - Near 3000: 2990-3009, 3011-3060, 3062-3064, 3067-3071.
  - Row 3000 has cosine 0.99999 between Base and CV, i.e. untouched.
- The CustomVoice speaker rows (2861, 2864, 2873, 2875, 2878, 3010, 3061, 3065, 3066) are x-vector-scale: norm **13.9-15.6**, pairwise cosine **0.90-0.96**. In Base the same rows are at init level (norm 0.022).
- Our kurisu vector (row 3000) has norm **17.0** and cosine **0.945-0.960** with vivian, ono_anna and serena.
- The speaker slot therefore works on small directional differences over a large common component. Two x-vectors of the *same actress* recorded through different EQ will probably be closer still, so measure that cosine before relying on it (§5.3).

---

## 4. Our trainer today: map of `train_sft.py`

- **Text** (TS:100-103): `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"` is tokenized and the last 5 ids are dropped. So `text_ids = [im_start, assistant, \n, body…]` and `n_text = 3 + n_body`. This equals `input_id[:, :3]` + `input_id[:, 3:-5]` in generate.
- **Length** (TS:111): `8 + n_text + n_codec`. It is used for bucketing and the VRAM probe.
- **Collate** (TS:123-192). Prefix length is 8 for every row:

  | positions | text channel | codec channel | masks |
  |---|---|---|---|
  | 0-2 | role (TS:154) | 0 | codec mask False |
  | 3-6 | `tts_pad` (TS:155) | `nothink think_bos think_eos 0` (TS:162-168) | **6 = speaker slot, codec mask False (TS:180)** |
  | 7 | `tts_bos` (TS:156) | `codec_pad` | |
  | 8 … 8+n_text-4 | body (TS:157) | `codec_pad` (TS:169) | |
  | 8+n_text-3 | `tts_eos` (TS:158) | `codec_pad` (TS:170) | |
  | 8+n_text-2 | `tts_pad` | `codec_bos` (TS:171) | |
  | 8+n_text-1 … +n_codec-1 | `tts_pad` | c0 (TS:172), c1-c15 in `codec_ids` (TS:177) | `codec_mask` True (TS:181); labels = c0 (TS:175) |
  | 8+n_text-1+n_codec | `tts_pad` | `codec_eos` (TS:173) | label = eos (TS:176) |

  Masks: text mask `[:8+n_text+n_codec]` (TS:160), codec mask `[3:…]` (TS:179), attention (TS:182).
- **`build_inputs`** (TS:304-326): `text_projection(text_embedding(ids)) * text_mask` (TS:313-317), plus `codec_embedding(ids) * codec_mask`. **`codec_emb[:, 6, :] = spk`** (TS:320) hard-codes the slot. Codebooks 1-15 are summed over `codec_mask` (TS:323-325).
- **Loss** (TS:343-379): the main loss uses `labels`, which `ForCausalLMLoss` shifts internally. The sub-talker reads `hidden[:, :-1][codec_mask[:, 1:]]` (TS:365).
  - Nothing in the prefix is labelled, so language, instruct and speaker tokens are pure conditioning. No loss masking needs to change.
- **Positions.** With an attention mask, the talker computes positions as `cumsum(mask) - 1` (MD:1693-1704, 1794-1795). Right-padded rows of different prefix lengths are therefore handled correctly.
- **Speaker** (TS:253-301): one locked vector from `ref_list.txt` (mean of the clip x-vectors, rescaled to their mean norm), cached as `spk_embedding.pt`.
- **Save** (TS:384-413):
  - `copytree(base_dir)` copies `config.json` and the tokenizer files from Base (TS:391-392).
  - It sets `tts_model_type="custom_voice"`, `spk_id={name: 3000}` and `spk_is_dialect={name: False}` (TS:394-398).
  - It writes the vector into row 3000 (TS:407-409).
- **Equivalence with generate().** For `language="Auto"`, no instruct and one speaker, TS's layout matches MD:2135-2227 position for position. What TS does not do is *anything* that depends on the row.

---

## 5. Patch plan (not applied)

### 5.0 One row builder for (a), (b) and (c)

Replace the fixed 8-slot layout with a per-row offset computed from the general formula in §1.3.

- Do **not** use PR #361's "uniform 6-slot width". Its Auto rows become `nothink think_bos PAD think_eos SPK pad`, a layout `generate()` never builds (MD:2136-2140).
- Batches are right-padded and positions come from the attention mask, so rows can have different prefix lengths.

```python
# --- CodedDataset.__init__ (new args: default_speaker, spk_index, p_auto, seed)
tc = config.talker_config
self.tc, self.lang_ids = tc, {k.lower(): v for k, v in tc.codec_language_id.items()}
self.spk_index, self.p_auto, self.rng = spk_index, p_auto, random.Random(seed)
...
lang = (row.get("language") or "auto").lower()
if lang != "auto" and lang not in self.lang_ids:
    raise ValueError(f"unknown language {lang!r}")
ins_ids = None
if row.get("instruct"):
    ins_ids = processor(text=f"<|im_start|>user\n{row['instruct']}<|im_end|>\n",
                        return_tensors="pt")["input_ids"]
    ins_ids = (ins_ids.unsqueeze(0) if ins_ids.dim() == 1 else ins_ids).long()   # NO [:-5]
spk = (row.get("speaker") or default_speaker).lower()
self.items.append({..., "language": lang, "instruct_ids": ins_ids, "spk": self.spk_index[spk]})
# lengths: worst case (k=6) so bucketing/probe stay conservative
self.lengths = [(0 if it["instruct_ids"] is None else it["instruct_ids"].shape[1])
                + 9 + (it["text_ids"].shape[1] - 3) + 3 + len(it["codes"]) for it in self.items]

# --- __getitem__: also return language / instruct_ids / spk

def _layout(self, it, lang):
    tc = self.tc
    o = 0 if it["instruct_ids"] is None else it["instruct_ids"].shape[1]
    think = ([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id] if lang == "auto" else
             [tc.codec_think_id, tc.codec_think_bos_id, self.lang_ids[lang], tc.codec_think_eos_id])
    spk_pos = o + 3 + len(think)           # 6 / 7 without instruct  (MD:2166-2172)
    return o, think, spk_pos, spk_pos + 2  # P = first body position: 8 / 9

def collate(self, batch, train=True):
    torch, cfg, tc = self.torch, self.config, self.tc
    rows = []
    for d in batch:
        lang = d["language"]
        if train and lang != "auto" and self.rng.random() < self.p_auto:
            lang = "auto"                                   # language dropout (5.1)
        rows.append((d, lang, *self._layout(d, lang)))
    t = max(P + (d["text_ids"].shape[1] - 3) + 3 + len(d["audio_codes"]) for d, _, _, _, _, P in rows)
    b = len(batch)
    # ... allocate input_ids / codec_ids / masks / labels exactly as now ...
    spk_pos_t = torch.zeros(b, dtype=torch.long); spk_idx_t = torch.zeros(b, dtype=torch.long)
    for i, (d, lang, o, think, sp, P) in enumerate(rows):
        tid, codes = d["text_ids"], d["audio_codes"]
        nb, nc = tid.shape[1] - 3, codes.shape[0]
        L = P + nb + 3 + nc
        if o:
            input_ids[i, :o, 0] = d["instruct_ids"][0]       # text only (MD:2079-2080)
        input_ids[i, o:o + 3, 0] = tid[0, :3]                # role
        input_ids[i, o + 3:P - 1, 0] = cfg.tts_pad_token_id
        input_ids[i, P - 1, 0] = cfg.tts_bos_token_id
        input_ids[i, P:P + nb, 0] = tid[0, 3:]
        input_ids[i, P + nb, 0] = cfg.tts_eos_token_id
        input_ids[i, P + nb + 1:L, 0] = cfg.tts_pad_token_id
        text_embedding_mask[i, :L] = True
        input_ids[i, o + 3:P, 1] = torch.tensor(think + [0, tc.codec_pad_id])
        input_ids[i, P:P + nb + 1, 1] = tc.codec_pad_id
        input_ids[i, P + nb + 1, 1] = tc.codec_bos_id
        input_ids[i, P + nb + 2:P + nb + 2 + nc, 1] = codes[:, 0]
        input_ids[i, P + nb + 2 + nc, 1] = tc.codec_eos_token_id
        codec_0_labels[i, P + nb + 2:P + nb + 2 + nc] = codes[:, 0]
        codec_0_labels[i, P + nb + 2 + nc] = tc.codec_eos_token_id
        codec_ids[i, P + nb + 2:P + nb + 2 + nc] = codes
        codec_embedding_mask[i, o + 3:L] = True
        codec_embedding_mask[i, sp] = False
        codec_mask[i, P + nb + 2:P + nb + 2 + nc] = True
        attention_mask[i, :L] = 1
        spk_pos_t[i], spk_idx_t[i] = sp, d["spk"]
    return {..., "spk_pos": spk_pos_t, "spk_idx": spk_idx_t}
```

With `o=0` and `lang="auto"`, this reproduces TS:154-182 index for index (P=8, `nb = n_text-3`, `L = 8+n_text+n_codec`).

`build_inputs` (replaces TS:320):

```python
spk_pos = batch["spk_pos"].to(device); spk_idx = batch["spk_idx"].to(device)
codec_emb = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_mask_e
codec_emb[torch.arange(codec_emb.shape[0], device=device), spk_pos] = spk_table[spk_idx].to(codec_emb.dtype)
```

Here `spk_table` is `[n_spk, 2048]` and replaces the single `spk`.

Other call sites:
- `evaluate()` calls `collate(..., train=False)` so validation is deterministic.
- `autotune_batch_size` needs no change; it still collates the longest rows.

### 5.1 (a) Per-row language id

- **Data.** Add `"language": "japanese"` to the JA rows and `"chinese"` to zh rows.
- **Language dropout `p_auto ≈ 0.3`.**
  - If every JA row carried `japanese`, the nothink (Auto) prefix would never be trained after the fine-tune, and the Space's current `language="Auto"` call would drift back toward the Base prior.
  - Dropping to Auto for about 30 % of rows keeps both prefixes tied to Kurisu.
  - Validation uses the row's own label.
- **Warm start.** Row `ja` (2058) is pretrained (norm 1.10) and our v2 has never touched it.
- **Unseen combinations.** `language="English"` with `speaker="kurisu"` was never seen in our fine-tune and falls back to Base's think+en behaviour. That is fine, but evaluate it.
- **Dialect override.** For every speaker with `spk_is_dialect` ≠ False, collate must mirror MD:2118-2122. With all values `False`, assert that instead.
- **Size:** about 25 changed lines (the row builder) + 5 (dropout) + 3 CLI args (`--p_auto_lang`).

### 5.2 (b) Per-row optional instruct

- Add `"instruct": "<fixed canonical string>"` to the conditioned rows. The string must be **byte-identical** in `app.py`; keep the strings in one shared constant table.
- No instruct dropout on channel-conditioned rows. Dropping it would teach the *default* voice to sound like Amadeus.
- Length bookkeeping: add `n_ins` (already in the row builder).
- For a wide prompt mix, either share one fixed string per condition, or vary paraphrases and also put style instructs on ordinary rows (see the design choice in §2.4).
- **LR.** At lr 5e-6 for one epoch the effect is likely weak (§2.4). Options:
  - a separate param group for `talker.model.layers.*.self_attn.*` and `text_projection` at about 2-5e-5;
  - 2-3 epochs;
  - the existing LoRA path at 1e-4.
- **Size:** about 10 lines on top of 5.0.

### 5.3 (c) Multiple speaker rows

```python
# CLI: --speakers "kurisu=/w/build/ref_list.txt,kurisu_amadeus=/w/build/ref_amadeus.txt,zh_dub=/w/build/ref_zh.txt"
specs = [s.split("=", 1) for s in args.speakers.split(",")]
spk_names = [n.lower() for n, _ in specs]
spk_table = torch.cat([speaker_embedding(model, Path(p), device, dtype, run_dir / f"spk_{n}.pt")
                       for n, p in specs], 0)                       # [n_spk, 2048]
cos = F.normalize(spk_table.float(), dim=-1); print("[speaker] cos\n", cos @ cos.T)
spk_index = {n: i for i, n in enumerate(spk_names)}
# save_checkpoint:
SPK_ROWS = [3000, 3001, 3002, 3003, 3004]      # untouched (norm≈0.02) in Base, CV and v2
talker_cfg["spk_id"] = {n: SPK_ROWS[i] for i, n in enumerate(spk_names)}
talker_cfg["spk_is_dialect"] = {n: False for n in spk_names}      # every key! (MD:2120)
for i, _ in enumerate(spk_names):
    weight[SPK_ROWS[i]] = spk_table[i].to(weight.dtype)
```

- Add `"speakers"` to the resume immutables (TS:555).
- Each speaker gets its own locked vector, mirroring CustomVoice inference (MD:2094-2101).
- **Optional learnable rows.** If the printed `cos(kurisu, kurisu_amadeus)` is above about 0.98, the slot cannot separate the two by much. In that case:
  - make `spk_table` an `nn.Parameter` initialised from the x-vectors;
  - put it in its own optimizer group (lr about 1e-4, weight decay 0);
  - save the learned rows.
  - About 8 more lines.
- **Size:** about 35 lines.

### 5.4 (d) Dedicated tag tokens: add tokens, no resize

```python
from tokenizers import AddedToken
TAGS = {"[sigh]": " sigh", "[gasp]": " gasp", "[breath]": " breath", "[laugh]": " laugh"}
tok = processor.tokenizer
assert len(tok) == 151676
tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in TAGS])
new_ids = tok.convert_tokens_to_ids(list(TAGS))                       # expect 151676..151679
assert max(new_ids) < config.talker_config.text_vocab_size            # 151936 -> no resize
emb = model.talker.model.text_embedding.weight
with torch.no_grad():                                                 # warm start from the English word
    for t, w in TAGS.items():
        emb[tok.convert_tokens_to_ids(t)] = emb[tok(w, add_special_tokens=False)["input_ids"]].mean(0)
# ids: " sigh"=30138, " breath"=11486, " laugh"=12550, " gasp"=[6819,79]
```

- **Order.** Add the tokens **before** `CodedDataset` tokenizes, then train on `text_tagged`.
- **Trade-off of the warm start.** `[sigh]` initially *is* the word "sigh", so the model will try to pronounce it until training overrides that. A random init is ignored at first instead, but learns more slowly. This is the same pronunciation risk as `（ため息）`.
- **Atomic matching.** Added tokens are matched before pre-tokenization, so `[sigh]` becomes one id whatever its neighbours.
- **LR for the new rows.** Adam's step size does not depend on gradient scale, so a gradient hook will not speed these rows up. To move them faster than lr 5e-6:
  - add a small `nn.Parameter` delta `[n_tags, 2048]`, applied in `build_inputs` where `input_ids[...,0]` is a tag id;
  - put it in its own group (about 1e-4);
  - fold it into `text_embedding` at save time.
  - About 15 lines.
- **Saving the tokenizer** is required: `processor.tokenizer.save_pretrained(out_dir)` in `save_checkpoint`, after the `copytree`. TS:391-392 currently copies Base's tokenizer files, which lack the new tokens. Verify the round trip by reloading `AutoProcessor.from_pretrained(out_dir, fix_mistral_regex=True)` in a fresh process and checking the ids. Different transformers versions on the pod and in the Space must agree.
- **`[pause]`:** keep P03's `"……"`. It is already a single native token (14053) and the script's own convention.
- **Data reality** (P03:31-37, 132):
  - The kept corpus has 148 `[sigh]`, 58 `[gasp]` and 16 `[breath]`.
  - `tagged_text()` keeps **only** `[sigh]`, `[gasp]` and `[pause]`. `[breath]` and `[laugh]` are already stripped, so P03 must change to use them.
  - Expected outcome: `[sigh]` will probably bind, `[gasp]` is borderline, and `[breath]`/`[laugh]` will not bind without more examples.
  - The Qwen team's own position (issue #9) is that there is no explicit non-verbal support; such sounds appear only implicitly through "hahaha" or "emmm".
- **Alternative with no tokenizer change:** substitute `[sigh]` → `<|object_ref_start|>` (and so on) by string replacement in both P03 and app. The tokens are atomic, but their rows carry LM priors.
- **Size:** about 25 lines, +15 for the delta parameter.

### 5.5 What changes in `app.py` and the saved `config.json`

- **`config.json`** (written by `save_checkpoint`):
  - `tts_model_type` stays `"custom_voice"`.
  - `spk_id` lists every speaker; `spk_is_dialect` has the **same keys**, all `false`.
  - `codec_language_id` stays unchanged (Base's 10 languages).
  - If a custom language key were ever added, its id must be below 3072 and taken from the untouched rows in §3.3. **Not recommended.**
- **Tokenizer files:** as in §5.4.
- **`app.py`:**
  - Add a **speaker** dropdown from `MODEL.get_supported_speakers()`. `verify_checkpoint` (APP:75-81) already handles multiple keys.
  - Add a **language** dropdown: Auto, Japanese, Chinese, English, … validated by `get_supported_languages()`. Replace the hard-coded `language="Auto"` (APP:154).
  - If instruct is used: a **condition** dropdown that maps to the exact canonical training strings, passed as `instruct=`. `generate_custom_voice` forwards it for 1b7 (QM:799-836, MD:2076-2080).
  - Tags need no code change: `validate_text` keeps brackets (APP:113-123). `_SPACE_NEAR_JA` (APP:20-21) removes spaces around a tag next to Japanese text, the same as P03's `normalise_text`. Add a hint listing the tags.
  - **Keep `non_streaming_mode=True`** (APP:155).
- **Size:** about 25 lines.

### 5.6 Verification (the risk control)

1. **Embedding-level equivalence test.** This one test retires the offset risk for (a), (b) and (c).
   - Load a saved checkpoint in **fp32**.
   - Monkeypatch `model.talker.generate` to raise with its `inputs_embeds` (it is called at MD:2272).
   - Call `wrapper.generate_custom_voice(text, speaker, language, instruct, non_streaming_mode=True)`.
   - Then build the same row with `collate([item], train=False)` + `build_inputs`, using `spk_table` read back from the saved rows.
   - Assert `allclose(atol=1e-5)` for positions `[0, P+nb+2)`, i.e. up to and including `tts_pad+codec_bos`.
   - Cover the matrix {Auto, Japanese} × {no instruct, instruct} × {kurisu, second speaker} × {plain text, text with `[sigh]`}.
   - About 60 lines, CPU-feasible.
2. **Tokenizer round trip** (§5.4).
3. **Probes before training:**
   - the instruct probe on the published checkpoint (§2.5);
   - Base `generate_voice_clone(x_vector_only_mode=True)` with Amadeus reference clips. If the x-vector alone reproduces the EQ, the speaker-row mechanism is strongly supported.
4. **Condition eval after training.** Generate the same 50 texts under each condition and compare spectral tilt against the corpus split (79 % of Amadeus above +5 dB vs 21 % of ordinary takes, P03:9-11). The default voice must stay near the ordinary distribution (leak check). Also check CER and similarity on the confirm suite against v2 (CER 0.030, sim 0.667).

### 5.7 Size and risk

| Item | Lines (approx.) | Main risk | Mitigation |
|---|---|---|---|
| Row builder + language (a) | 30 | Offset arithmetic | Equivalence test; reduces to TS exactly for Auto |
| Instruct (b) | 10 (+10 LR group) | Weak effect at lr 5e-6 × 1 epoch; default-voice leakage | Probe first; tilt eval; higher LR on attention |
| Multi-speaker (c) | 35 (+8 learnable) | Amadeus x-vector too close to kurisu | Print cosine; learnable rows |
| Tags (d) | 25 (+15 delta) | Too few examples; tokenizer save/load mismatch | Train only sigh and gasp; reload test |
| `app.py` | 25 | Instruct strings drifting from training | Shared constant table |
| Equivalence test | 60 | none | none |
| **Total** | **about 150-200** | Moderate overall. The only silent failure mode is offsets, which the test closes. | |

---

## 6. Recommended mechanism per condition

| Condition | Mechanism | Why |
|---|---|---|
| (a) Japanese and other language ids | **Native think-mode codec language id**, per-row variable prefix, `p_auto ≈ 0.3` | The exact `generate()` path. `ja`/`zh` rows are pretrained warm starts. |
| (b) Chinese dub actress | **A new speaker row** `zh_dub` (3001) with its own locked x-vector, rows labelled `chinese` | It is a different speaker, so it must have its own slot. |
| (b) Synthetic Chinese in Kurisu's voice | Speaker `kurisu` + `language: chinese` | Keep the volume small and the quality high: the speaker is shared, so synthetic artefacts transfer to her. |
| (c) Amadeus and telephone | **Primary: dedicated speaker rows** `kurisu_amadeus`, `kurisu_phone` (3002, 3003). All Amadeus takes go to `kurisu_amadeus`; the fiction puts every Amadeus line through a device, and tilt is only a proxy. Phone takes go by the `[digital]` tag. **No device-processed take ever goes into the default `kurisu` row.** Takes below +5 dB tilt may be dropped instead, but never relabelled as `kurisu`. **Secondary: a fixed instruct string** if composition (e.g. zh_dub through a phone) matters. | The speaker slot is the strongest, already-trained conditioning path and needs no inference change. Instruct is "unstable" on 12Hz Base, and our training is about 10× lighter than CV's. |
| (c) Not recommended | Dialect override; repurposing the language slot for a channel | The dialect override only fires for chinese/auto (MD:2118). The language slot cannot be shared, so Japanese + Amadeus could not both be specified. |
| (d) `[sigh]`, `[gasp]` (`[breath]`, `[laugh]` only with more data) | **Added atomic tokens** at 151676+ (no resize), warm-started from the English word embedding, higher-LR delta, tokenizer saved in the checkpoint | Plain-text tags split into 3-5 context-dependent pieces. Reserved specials carry LM priors. |
| `[pause]` | Keep `……` (token 14053) | Native and already single-token |

---

## 7. Upstream changes after 0.1.1 (as of 2026-10-28)

**Release and commits:**
- Releases: 0.1.0 on 2026-02-05, 0.1.1 on 2026-02-06. There are no git tags.
- Only one commit after 0.1.1: **`022e286` (2026-03-17) "fix finetuning bug"**, `hidden_states[codec_mask[:, 1:]]` → `[codec_mask[:, :-1]]` in UP-sft. TS's `--label_alignment official` already uses the latter.
- The `qwen_tts/` package on main is unchanged from the wheel.
- `finetuning/dataset.py` still reads `language` and discards it (UP-ds:126). Its collate still hard-codes nothink and speaker slot 6 (UP-ds:180-201). UP-sft still writes `spk_id={name: 3000}` (UP-sft:136-141).

**Open PRs that touch these features (none merged):**
- **#361** (2026-08-28) "actually use the language field". Uses a uniform 6-slot layout. **Bug:** Auto rows become `nothink think_bos PAD think_eos SPK pad`, which `generate()` never produces. Language rows are correct (speaker slot 7, body 9).
- **#384** (2026-10-17) multi-speaker SFT plus the text_projection, double-shift and sub-talker fixes.
  - Optional `speaker` field; rows are `3000 + index` in sorted order; every `spk_is_dialect` is False.
  - It keeps the *first* per-step encoder embedding seen per speaker rather than a locked mean, and its slot is still fixed at 6.
  - No language or instruct support.
- **#281** pre-wires instruct for `generate_voice_clone` (for the 25Hz model).
- Fixes that TS already contains: #278 / #336 / #188 (text_projection, label alignment), #351 (speaker-encoder decay).
- Also open: #373 and #379 (attention implementation, NaN above 8,192 frames), #295, #222 and #232 (keep `speaker_encoder` in checkpoints).

**Issues:**
- **#323** (Bengali): the maintainer-style answer from a community user describes think vs nothink correctly, but has two errors:
  - it proposes codec id **4206**, outside the 3072-row codec vocabulary;
  - it says the speaker placeholder "is not needed when language is specified", but `generate()` always inserts SPK after `think_eos` (MD:2170-2172).
- **#25:** Base instruct is not supported ("relatively unstable").
- **#14:** a 25Hz VoiceEditing model (clone + instruct) is planned, with no date.
- **#9:** no explicit non-verbal tags.
- **#248:** CustomVoice instruct handles emotion and prosody, not dialect.
- **#343:** onset timbre instability after a LoRA fine-tune; points to #351.

---

## Appendix: raw numbers

- **Codec embedding norms (Base):** codebook rows 0-2047 have mean 0.73 (0.57-1.04). Control rows 2048-3071 have median 0.023: 816 rows are below 0.156, and the trained ones range 0.26-1.64 (bos 1.64, eos 1.30, pad 0.93).
- **Rows that differ between Base and CV:** all 3072 (by about 1e-3 to 7e-3 on the named control rows). Speaker rows jump from 0.022 to about 15.
- **Text embedding tail:** in Base, CV and VD, rows 151676-151935 have norm 0.298-0.321 and pairwise cosine 0.000. The added rows 151643-151675 have norm 0.13-0.70. Regular rows 0-1999 have mean norm 0.61.
- **Our v2 checkpoint:**
  - text_embedding rows 0-1999 drifted 0.095 %;
  - codec rows `nothink` 0.62 %, `pad` 0.40 %, `bos` 0.19 %, and `think`/`ja`/`en`/`zh` 0;
  - row 3000 has norm 17.04.
- **Scratch artefacts** (scripts and fetched data, not part of the project) were kept in a session scratch folder and are not published.
