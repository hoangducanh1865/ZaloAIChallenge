#!/usr/bin/env bash
# Zero-shot baselines on the first LIMIT train samples (labelled, never trained on), all runs sharing the GPU; per
# sample outputs on Drive, a re-run resumes. A spec is <model config>[:think]:
#   (setsid nohup bash scripts/a100_baselines.sh gemma4_e2b qwen25_omni_3b gemma4_e4b:think > <log on Drive> 2>&1 < /dev/null &)
# Times per sample are inflated by the sharing: measure latency with a run alone on the GPU (scripts/a100_eval.sh).
#
# Author: Duc Anh Hoang
source /root/.hf_env 2> /dev/null || true   # HF_TOKEN; first, before any set -x
set -x
LIMIT=${LIMIT:-100}
cd /content/drive/MyDrive/it/zalo/code/ZaloAIChallenge
LOG=/content/drive/MyDrive/it/zalo/logs/baselines; mkdir -p $LOG
export HF_HOME=${HF_HOME:-/content/hf}
for spec in "$@"; do
  m=${spec%%:*}; t=false; tok=512
  [[ $spec == *:think ]] && t=true && tok=2048
  python -u run.py model=$m agent.enable_thinking=$t agent.max_new_tokens=$tok data=train data.limit=$LIMIT \
    > $LOG/${m}_think$t.log 2>&1 &
done
wait
grep -H "SCORE" $LOG/*.log
echo "BASELINES EXIT"
