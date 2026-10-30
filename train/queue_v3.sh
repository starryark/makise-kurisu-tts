#!/bin/bash
# v3 experiment queue (SELECTION_RULE.md, v3 + Amendment A). Each ablation changes one
# thing relative to its parent, trains with v2's schedule (lr 5e-6, batch 32, 2-epoch
# cosine) and stops at 1.0 epoch - where v2's winner sat - so one checkpoint per run is
# scored (S=2 sampling seeds, a screen). The final run keeps 0.75/1.0/1.25-epoch
# checkpoints and scores them at S=4, then the pick is re-tested at seed base 26000.
#   usage: queue_v3.sh <stage> ...
#   stages: e0 baselines pseudo E1a E1b E2 E3 E3f E4 E5 E5b E6 E6b E7 final retest
set -euo pipefail
export KURISU_WORK=/workspace/kurisu HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1
export KURISU_V2_REV=92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57
W=/workspace/kurisu
cd $W/code
V2="starrydark/Kurisu_Qwen3_TTS@$KURISU_V2_REV"
BUILD=$W/build_v3
SPK1="kurisu=$BUILD/ref_kurisu.txt"
SPK3="kurisu=$BUILD/ref_kurisu.txt,amadeus=$BUILD/ref_amadeus.txt,phone=$BUILD/ref_phone.txt"
SPK4="kurisu=$BUILD/ref_kurisu.txt,amadeus=$BUILD/ref_amadeus.txt,amadeus_anime=$BUILD/ref_amadeus_anime.txt,phone=$BUILD/ref_phone.txt"
AUX=$W/replay/aux_speakers.json
TOK=$W/code/new_tokens_v3.json
TOK_SFX=$W/code/new_tokens_v3_sfx.json
FREEZE="model.text_embedding,text_projection"

mix () {   # mix <name> <mix_v3.py flags...>
  local name=$1; shift
  python mix_v3.py --out $W/qwen_v3/exp/$name "$@" > $W/logs/mix_$name.log
  tail -n 30 $W/logs/mix_$name.log
}

train () { # train <name> <data> <seed> <lang_trained true|false> <token file|-> <extra train_sft.py flags...>
  local name=$1 data=$2 seed=$3 lang=$4 tok=$5; shift 5
  if compgen -G "$W/runs/$name/checkpoint-step-*/kurisu_v3.json" > /dev/null; then
    echo "[queue] $name already trained - skipped (rm -rf $W/runs/$name to retrain)"; return 0
  fi
  rm -rf "$W/runs/$name"          # partial run from a crash: start clean
  local tokargs=()
  [ "$tok" != "-" ] && tokargs=(--new_tokens "$tok")
  python train_sft.py \
    --train_jsonl $W/qwen_v3/exp/$data/train_with_codes.jsonl \
    --val_jsonl $W/qwen_v3/exp/$data/val_with_codes.jsonl \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --speaker_name kurisu --attn sdpa \
    --mode full --full_dtype fp32 --optimizer adamw --lr 5e-6 --weight_decay 0.01 \
    --epochs 2 --batch_size 16 --effective_batch 32 \
    --max_val_batches 1000 --eval_every 50 --log_every 10 --keep_last 0 \
    --vram_fraction 0.95 --tensorboard --seed $seed \
    --out $W/runs/$name "${tokargs[@]}" "$@" 2>&1 | tee $W/logs/train_$name.log
  local toks="[]"
  [ "$tok" != "-" ] && toks=$(python -c "import json,sys; print(json.dumps(list(json.load(open(sys.argv[1],encoding='utf-8')))))" "$tok")
  for ck in $W/runs/$name/checkpoint-step-*; do
    echo "{\"lang_trained\": $lang, \"tokens\": $toks, \"run\": \"$name\"}" > $ck/kurisu_v3.json
  done
  rm -rf $W/runs/$name/checkpoint-epoch-* $W/runs/$name/state.pt   # unscored; full-mode resume is refused anyway
}

score () { # score <name> <seeds> [extra p06 flags]: every checkpoint of a run
  local name=$1 seeds=$2; shift 2
  local args=()
  for ck in $W/runs/$name/checkpoint-step-*; do args+=(--ckpt "$ck"); done
  python p06_eval_v3.py --name v3 --skip_existing --seeds $seeds "${args[@]}" "$@" \
      2>&1 | tee -a $W/logs/eval_$name.log
  echo "STAGE_DONE $name"
}

ablate () { # ablate <name> <data> <lang_trained> <token file|-> <extra train flags...>
  local name=$1 data=$2 lang=$3 tok=$4; shift 4
  train $name $data 20260821 $lang $tok --stop_after 1.0 "$@"
  score $name 2
}

for stage in "$@"; do
  case "$stage" in
    e0)        python p10_e0.py 2>&1 | tee $W/logs/e0.log; echo "STAGE_DONE e0" ;;
    baselines) python p06_eval_v3.py --name v3 --skip_existing --seeds 2 --ckpt real --ckpt $V2 \
                   2>&1 | tee $W/logs/eval_baselines.log
               python p07_select_v3.py --suite v3 --ref "$V2" --check_a1 \
                   2>&1 | tee $W/logs/a1_check.log
               echo "STAGE_DONE baselines" ;;
    pseudo)    python p11_pseudo_zh.py 2>&1 | tee $W/logs/pseudo.log
               python p04_codes_v3.py 2>&1 | tee -a $W/logs/codes_v3.log
               echo "STAGE_DONE pseudo" ;;
    E1a)  mix E1 --v2_ids v2_kept_ids.txt
          ablate E1a E1 false - --speakers $SPK1 ;;
    E1b)  train E1b E1 20261028 false - --speakers $SPK1 --stop_after 1.0
          score E1b 2 ;;
    E2)   mix E2 --v2_ids v2_kept_ids.txt --language japanese
          ablate E2 E2 true - --speakers $SPK1 --p_auto_lang 0.25 ;;
    E3)   mix E3 --v2_ids v2_kept_ids.txt --language japanese --replay --replay_share 0.15
          ablate E3 E3 true - --speakers $SPK1 --aux_speakers $AUX --p_auto_lang 0.25 ;;
    E3f)  ablate E3f E3 true - --speakers $SPK1 --aux_speakers $AUX --p_auto_lang 0.25 \
              --freeze "$FREEZE" ;;
    E4)   mix E4 --v2_ids v2_kept_ids.txt --language japanese --replay --replay_share 0.15 \
              --pseudo_share 0.06
          ablate E4 E4 true - --speakers $SPK1 --aux_speakers $AUX --p_auto_lang 0.25 \
              ${E4_FREEZE:+--freeze "$FREEZE"} ;;
    E3b)  # Amendment D: matched-regime replication (training seed 20261028)
          train E3b E3 20261028 true - --speakers $SPK1 --aux_speakers $AUX --p_auto_lang 0.25               --stop_after 1.0
          score E3b 2 ;;
    E4b)  train E4b E4 20261028 true - --speakers $SPK1 --aux_speakers $AUX --p_auto_lang 0.25               --stop_after 1.0
          score E4b 2 ;;
    E5)   mix E5 --v2_ids v2_kept_ids.txt --speakers kurisu,amadeus,amadeus_anime,phone \
              --phone_aug --phone_repeat 4
          ablate E5 E5 false - --speakers $SPK4 ;;
    E5b)  # Amendment B1: no amadeus_anime row (its vector cannot carry that sound, E0)
          mix E5b --v2_ids v2_kept_ids.txt --speakers kurisu,amadeus,phone               --phone_aug --phone_repeat 4
          ablate E5b E5b false - --speakers $SPK3 ;;
    E6b)  # Amendment B2: E6 without upsampling event rows
          mix E6b --v2_ids v2_kept_ids.txt --text v3 --tag_only --event_repeat 1
          ablate E6b E6b false $TOK --speakers $SPK1 ;;
    E6)   mix E6 --v2_ids v2_kept_ids.txt --text v3 --tag_only --event_repeat 3
          ablate E6 E6 false $TOK --speakers $SPK1 ;;
    E7)   mix E7 --v2_ids v2_kept_ids.txt --text v3 --tag_only --event_repeat 3 --sfx
          ablate E7 E7 false $TOK_SFX --speakers $SPK1 ;;
    final)
      # FINAL_MIX / FINAL_TRAIN / FINAL_TOK / FINAL_LANG are set from the adopted
      # ablations (decision recorded in logs/final_decision.txt before this runs).
      test -s $W/logs/final_decision.txt
      mix final ${FINAL_MIX:?}
      case "${FINAL_LANG:?}" in true|false) ;; *) echo "FINAL_LANG must be true|false"; exit 1 ;; esac
      eval "ftrain=(${FINAL_TRAIN:?})"; eval "ftok=${FINAL_TOK:--}"
      for seed in 20260821 20261028; do
        train final_s$seed final $seed $FINAL_LANG "$ftok" "${ftrain[@]}" \
            --stop_after 1.25 --save_at 0.75,1.0
        # v2 is generated at every base the candidates use (A2 pairs common bases)
        python p06_eval_v3.py --name v3 --skip_existing --seeds 4 --ckpt $V2 \
            2>&1 | tee -a $W/logs/eval_baselines.log
        score final_s$seed 4
      done
      cands=(); for ck in $W/runs/final_s*/checkpoint-step-*; do cands+=(--cand "$ck"); done
      python p07_select_v3.py --suite v3 --ref "$V2" --e1 E1a,E1b "${cands[@]}" \
          2>&1 | tee $W/logs/select_final.log
      echo "STAGE_DONE final" ;;
    retest)
      # A2: the pick, re-tested at fresh seeds against v2 at the same seeds
      python p06_eval_v3.py --name v3 --skip_existing --seed_bases 26000 --ckpt $V2 --ckpt ${PICK:?} \
          2>&1 | tee $W/logs/eval_retest.log
      python p07_select_v3.py --suite v3 --ref "$V2" --seed_bases 26000 \
          --cand ${PICK} 2>&1 | tee $W/logs/select_retest.log
      echo "STAGE_DONE retest" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo QUEUE_DONE
