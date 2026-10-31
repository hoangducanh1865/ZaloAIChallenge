#!/usr/bin/env bash
# Small two-stage fine-tuning experiment for Gemma-4-E2B on the A100, evaluated on the held-out train[:100]:
#   A  stage 1 (ASR, ViMD subset) -> ASR probe of the adapter
#   B  stage 2 (agent, LifeMate train[100:]) from the base model          (ablation: no stage 1)
#   C  stage 2 from the stage-1 adapter
#   then the agent score (audio input) of B and C, next to the zero-shot 4.80.
# Checkpoints, logs and results on Drive; every step resumes.
#   (setsid nohup bash scripts/a100_finetune.sh > /content/drive/MyDrive/it/zalo/logs/a100_finetune.log 2>&1 < /dev/null &)
#
# Author: Duc Anh Hoang
source /root/.hf_env 2> /dev/null || true   # HF_TOKEN; first, before any set -x
set -x
cd /content/drive/MyDrive/it/zalo/code/ZaloAIChallenge
export HF_HOME=${HF_HOME:-/content/hf}
LOG=/content/drive/MyDrive/it/zalo/logs/finetune; mkdir -p $LOG
CK=/content/drive/MyDrive/it/zalo/ckpt; RUNS=/content/drive/MyDrive/it/zalo/runs
MODEL=gemma4_e2b
pip install -q -r requirements.txt > $LOG/pip.log 2>&1 &
python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-E2B-it')" > $LOG/fetch.log 2>&1 &
wait

python -u train.py model=$MODEL stage=asr > $LOG/stage1.log 2>&1 &
S1=$!
python -u train.py model=$MODEL stage=agent +run_suffix=_from_base > $LOG/stage2_from_base.log 2>&1 &
S2B=$!
wait $S1
python -u scripts/asr_probe.py model=$MODEL +model.adapter=$CK/${MODEL}_asr/adapter data=train data.limit=100 \
  out_dir=$RUNS/asr_${MODEL}_ft > $LOG/asr_probe_ft.log 2>&1 &
python -u train.py model=$MODEL stage=agent init_adapter=$CK/${MODEL}_asr/adapter +run_suffix=_from_asr \
  > $LOG/stage2_from_asr.log 2>&1
wait $S2B
for v in from_base from_asr; do  # one gold call per model turn in training: allow up to 8 turns
  python -u run.py model=$MODEL +model.adapter=$CK/${MODEL}_agent_$v/adapter agent.max_steps=8 +run_suffix=_ft_$v \
    data=train data.limit=100 > $LOG/eval_$v.log 2>&1 &
done
wait
grep -h "TRAIN DONE\|ASR \|SCORE\|LATENCY" $LOG/*.log
echo "A100 FINETUNE EXIT"
