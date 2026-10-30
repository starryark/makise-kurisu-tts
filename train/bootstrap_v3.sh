#!/bin/bash
# Fresh pod -> environment, models, corpus (JA + ZH/EN dub anchors), then the v3 data
# chain up to coded experiment sources. Expects /workspace/kurisu/code (this dir),
# /workspace/kurisu/src/{asr.jsonl,labels_v3.jsonl} uploaded from the local machine
# (ASR is not re-run: same audio, same models as v2), and the HF token at
# /workspace/hf/token (HF_HOME) - never printed.
set -euo pipefail
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 KURISU_WORK=/workspace/kurisu PYTHONUNBUFFERED=1
mkdir -p /workspace/kurisu/logs /workspace/kurisu/src
cd /workspace/kurisu
[ -s src/asr.jsonl ] && [ -s src/labels_v3.jsonl ] || { echo 'upload src/asr.jsonl and src/labels_v3.jsonl first'; exit 1; }

# 0. Fail fast on a bad GPU.
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); print('gpu ok', torch.cuda.get_device_name(0), float((x@x).abs().mean()))"

# 1. Environment (v2's, minus UTMOS; plus pypinyin for the accent metrics).
python -m pip install -q --upgrade pip
pip install -q --ignore-installed blinker
pip install -q "qwen-asr==0.0.6" hf_transfer pyopenjtalk-plus speechbrain==1.0.3 "numpy<2.3" \
    soundfile pyloudnorm tensorboard pyarrow pypinyin
pip install -q --no-deps "qwen-tts==0.1.1"
pip install -q onnxruntime einops sox
python -c "import torch, transformers, qwen_asr, qwen_tts, pyopenjtalk, pypinyin, speechbrain; print('env ok', torch.__version__, transformers.__version__)"
echo STAGE_ENV_DONE

# 2. Models (background) while the corpus downloads.
(cd /workspace/kurisu/code; for m in Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
          Qwen/Qwen3-TTS-Tokenizer-12Hz speechbrain/spkrec-ecapa-voxceleb \
          speechbrain/lang-id-voxlingua107-ecapa charsiu/zh_xlsr_fc_10ms charsiu/tokenizer_zh_pinyin \
          mispeech/ced-base; do
   python -c "from huggingface_hub import snapshot_download as s; import sys; s(sys.argv[1]); print('model ok', sys.argv[1])" "$m"
 done
 python -c "import accent_metrics as A, ced_tagger as C; from huggingface_hub import snapshot_download as s; [s(r, revision=v) for r, v in ((A.LID_REPO, A.LID_REV), (A.CHARSIU_REPO, A.CHARSIU_REV), (A.CHARSIU_TOK_REPO, A.CHARSIU_TOK_REV), (C.CED_REPO, C.CED_REV))]; print('model ok pinned')"
 python -c "from huggingface_hub import snapshot_download as s; s('starrydark/Kurisu_Qwen3_TTS', revision='92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57'); print('model ok v2')"
) > logs/models.log 2>&1 &
MODELS_PID=$!

# 3. Corpus: parallel direct resolves (the hub client crawls on 10k small files).
cd src
cat /workspace/kurisu/code/ja_files.txt /workspace/kurisu/code/zhen_files.txt | tr -d '\r' | sort -u > list.txt
TOKEN=$(cat /workspace/hf/token)
xargs -P 64 -I{} sh -c '[ -s hf/{} ] || { mkdir -p "hf/$(dirname {})" && curl -sfL --retry 5 -H "Authorization: Bearer '"$TOKEN"'" -o "hf/{}" "https://huggingface.co/datasets/starrydark/Kurisu_Voice/resolve/main/{}" || echo "FAIL {}"; }' < list.txt
unset TOKEN
missing=$(while read -r f; do [ -s "hf/$f" ] || echo "$f"; done < list.txt | wc -l)
[ "$missing" -eq 0 ] || { echo "$missing corpus files missing - rerun bootstrap"; exit 1; }
echo "wavs: $(find hf -name '*.wav' | wc -l)"
cd ..
wait $MODELS_PID
grep -c "model ok" logs/models.log
echo STAGE_DOWNLOAD_DONE

# 4. Data chain.
cd code
python p01_fetch.py
python p03_build_v3.py > ../logs/build_v3.log 2>&1; tail -n 40 ../logs/build_v3.log
echo STAGE_BUILD_DONE
python p03c_augment.py --meta $KURISU_WORK/src/ja_meta.jsonl --wav_dir $KURISU_WORK/build_v3/wavs \
    --kept $KURISU_WORK/build_v3/kept.jsonl --out $KURISU_WORK/augment \
    --phone_narrowband_share 0 --sfx_classes footsteps,ringtone,rain,typing,traffic \
    > ../logs/augment.log 2>&1 || echo "WARN augment failed - E5/E7 will refuse to mix (see logs/augment.log)"
tail -n 25 ../logs/augment.log
echo STAGE_AUGMENT_DONE
# optional sources must not stop the chain: the Japanese codes below are needed by every run
python p05_replay.py > ../logs/replay.log 2>&1 || echo "WARN replay failed or short - E3/E4 will refuse to mix (see logs/replay.log)"
tail -n 20 ../logs/replay.log
echo STAGE_REPLAY_DONE
python p04_codes_v3.py > ../logs/codes_v3.log 2>&1; tail -n 20 ../logs/codes_v3.log
echo STAGE_CODES_DONE
