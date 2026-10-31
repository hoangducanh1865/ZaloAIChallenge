#!/usr/bin/env bash
# Zero-shot evaluation of one model on the A100, thinking off and on. Outputs and logs on Drive, saved per sample; a
# re-run resumes. Run from anywhere:
#   (setsid nohup bash scripts/a100_eval.sh gemma4_e4b > /content/drive/MyDrive/it/zalo/logs/a100_gemma4_e4b.log 2>&1 < /dev/null &)
# 1. Latency: the same LAT_N train samples per mode, each mode alone on the GPU (sharing it would inflate the times);
#    kept as latency_alone.json in the run folder.
# 2. Scores, the runs sharing the GPU: train TRAIN_LIMIT samples for both modes (resumes after step 1), public_test
#    with thinking off (submission.jsonl for the leaderboard); THINK_PUBLIC=1 adds public_test with thinking on.
#
# Author: Duc Anh Hoang
source /root/.hf_env 2> /dev/null || true   # HF_TOKEN; first, before any set -x
set -x
MODEL=${1:-gemma4_e4b}
LAT_N=${LAT_N:-50}
TRAIN_LIMIT=${TRAIN_LIMIT:-200}
cd /content/drive/MyDrive/it/zalo/code/ZaloAIChallenge
LOG=/content/drive/MyDrive/it/zalo/logs/${MODEL}; mkdir -p $LOG
RUNS=/content/drive/MyDrive/it/zalo/runs
export HF_HOME=${HF_HOME:-/content/hf}   # local disk: HF download takes about 1 min, Drive sync between runtimes lags
pip install -q -r requirements.txt > $LOG/pip.log 2>&1 &
python -c "from huggingface_hub import snapshot_download as s; from omegaconf import OmegaConf as O; \
s(O.load('configs/model/${MODEL}.yaml').model_id)" > $LOG/fetch.log 2>&1 &
wait
nvidia-smi --query-gpu=name,memory.total --format=csv

run() {  # run <thinking true|false> <hydra overrides...>
  local t=$1; shift
  local tok=512; [[ $t == true ]] && tok=2048
  python -u run.py model=$MODEL agent.enable_thinking=$t agent.max_new_tokens=$tok "$@"
}

for T in false true; do
  run $T data=train data.limit=$LAT_N > $LOG/latency_think$T.log 2>&1
  cp $RUNS/${MODEL}_think${T^}_train/latency.json $RUNS/${MODEL}_think${T^}_train/latency_alone.json   # run folder: thinkTrue / thinkFalse
done

run false data=public_test > $LOG/public_test_thinkfalse.log 2>&1 &
run false data=train data.limit=$TRAIN_LIMIT > $LOG/train_thinkfalse.log 2>&1 &
run true data=train data.limit=$TRAIN_LIMIT > $LOG/train_thinktrue.log 2>&1 &
[[ ${THINK_PUBLIC:-0} == 1 ]] && run true data=public_test > $LOG/public_test_thinktrue.log 2>&1 &
wait
grep -H "SCORE\|LATENCY" $LOG/*.log
echo "A100 JOB EXIT $?"
