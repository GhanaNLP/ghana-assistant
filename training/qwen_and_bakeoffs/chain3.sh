#!/bin/bash
# light deployable navigation model: Qwen3.5-0.8B, full train set, router mode (answer-only)
cd "$(dirname "$0")"; source .venv-unsloth/bin/activate; mkdir -p results
echo "=== train 0.8B full $(date +%T)" >> chain3.log
python -u train_unsloth.py --model unsloth/Qwen3.5-0.8B --out runs/nav08_full --n 0 --epochs 1 --bs 16 --accum 1 --answer_only --save_steps 2000 > logs_nav08_full.train 2>&1 && echo "ok train" >> chain3.log || echo "FAILED train" >> chain3.log
for sp in test_area test_pair; do
  echo "=== eval $sp $(date +%T)" >> chain3.log
  python -u eval_reason.py --mode router --adapter runs/nav08_full --split $sp --n 500 --bs 32 --out results/nav08_full_$sp.json > logs_nav08_eval_$sp.log 2>&1 && echo "ok $sp" >> chain3.log || echo "FAILED $sp" >> chain3.log
done
echo finished >> chain3.log
