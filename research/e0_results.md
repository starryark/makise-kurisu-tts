# E0 results: where the Mandarin accent comes from (2026-10-29, pod run)

**Setup.**
- 80 Mandarin + 50 English probes, 2 sampling seeds each.
- Upstream decoding defaults, non-streaming.
- Scores:
  - **M1** = VoxLingua107 log P(ja) − log P(zh). Lower means more native Mandarin.
  - **SER(shared)**: charsiu toneless syllable error rate on characters shared between Japanese and Chinese.
  - **CER**: Qwen3-ASR (forced zh) CER.
  - **sim**: ECAPA similarity to the real-Kurisu Japanese centroid.

## Mandarin

| Cond | What | M1 | SER shared | CER | sim |
|---|---|---|---|---|---|
| D | Base + native Mandarin reference (native anchor) | −22.6 | 0.048 | 0.013 | 0.227 |
| A | Base + Kurisu speaker vector, Auto | −19.3 | 0.065 | 0.061 | 0.571 |
| B | Base + Kurisu speaker vector, `language=Chinese` | −19.9 | 0.057 | 0.044 | 0.568 |
| G25 | v2 weights blended 25% toward v2 (75% Base), Chinese | −18.6 | 0.084 | 0.063 | 0.592 |
| G50 | 50% blend, Chinese | −17.6 | 0.095 | 0.033 | 0.628 |
| G75 | 75% blend, Chinese | −15.6 | 0.117 | 0.049 | 0.623 |
| F | v2, `language=Chinese` | −13.0 | 0.169 | 0.056 | 0.626 |
| R | real Chinese-dub recordings (other texts; acted, with music) | −13.5 | 0.227 | 0.038 | 0.246 |
| J | Qwen's own Japanese CustomVoice speaker (ono_anna), Chinese | −9.9 | 0.237 | 0.250 | 0.316 |
| E | v2, `language=Auto` (what the Space does today) | −6.6 | 0.289 | 0.154 | 0.633 |
| H | ono_anna, `language=Japanese` on Chinese text (wrong-language anchor) | +13.3 | 0.781 | 0.850 | — |
| C | Base, voice-clone ICL from a Japanese Kurisu take | +14.0 | 0.755 | 0.696 | 0.558 |

## Findings

1. **The accent is introduced by the fine-tune, not by Kurisu's voice.** The Base model driven by Kurisu's own speaker vector speaks near-native Mandarin (B: −19.9 vs native −22.6). v2 moves that to −13.0 / −6.6.
2. **The Space's `language="Auto"` makes it worse.**
   - Passing `language="Chinese"` to v2 alone halves the shared-character error (0.289 → 0.169) and cuts CER from 0.154 to 0.056.
   - Free fix for the deployed v2: pick the language from the script of the text.
3. **Weight blending is a free serving lever.** G50 keeps v2's similarity in Chinese (0.628 vs 0.626) with much less accent (M1 −17.6, SER 0.095).
4. **Qwen's own Japanese preset voice is accented in Mandarin too** (J: −9.9). The problem is general to single-language SFT, as the literature said.
5. **Voice-clone (ICL) from a Japanese take reads Chinese as Japanese** (C). Never use ICL from a Japanese reference for Chinese.
6. **The x-vector alone carries the device EQ** (Base + row vector, Japanese text), measured as projection onto the real-takes shift:
   - Amadeus (game): 1.04, tilt +11.4 dB.
   - phone: 1.22, tilt +18.3 dB.
   - Amadeus (anime): −0.43, its takes being a mix of clean and filtered.
   - So speaker rows are a sound mechanism for game Amadeus and phone.

## Metric validity (A4)

| Metric | Valid? | Evidence |
|---|---|---|
| zh M1 | valid | d(D,C) 6.1, monotone G25→F, seed reliability 0.75 |
| zh SER(shared) | valid | d 4.9, monotone, reliability 0.60 |
| zh SER (all), TER | invalid | not monotone |
| en M1 | passes the A4 criteria | d 2.8, monotone, reliability 0.62 |

English M1 passes on paper, but the rule pre-registered "no English claim", and that stands.

## A10 gate

M1(B) = −19.94 ≤ (M1(F) + M1(D)) / 2 = −17.81, so the pseudo-Mandarin route (p11 → E4) runs.
