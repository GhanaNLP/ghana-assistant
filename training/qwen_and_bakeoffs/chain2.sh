#!/bin/bash
cd "$(dirname "$0")"; source .venv-unsloth/bin/activate; mkdir -p results
echo "=== train 27B answer-only $(date +%T)" >> chain2.log
python -u train_unsloth.py --model unsloth/Qwen3.8-27B --out runs/q38_ans_10k --n 10000 --epochs 1 --bs 8 --accum 2 --answer_only --save_steps 300 > logs_q38_ans.train 2>&1 && echo "ok train 27B" >> chain2.log || echo "FAILED train 27B" >> chain2.log
for spec in "rt_Qwen3.5-0.8B|runs/rt_Qwen3.5-0.8B" "rt_Qwen3.5-2B|runs/rt_Qwen3.5-2B" "rt_Qwen3.5-4B|runs/rt_Qwen3.5-4B" "rt_Qwen3.5-9B|runs/rt_Qwen3.5-9B" "q38_nav_10k|runs/q38_nav_10k" "q38_ans_10k|runs/q38_ans_10k" "base27b|"; do
  n=${spec%%|*}; ad=${spec##*|}; echo "=== eval $n $(date +%T)" >> chain2.log
  if [ -n "$ad" ]; then A="--adapter $ad"; else A=""; fi
  python -u eval_reason.py --mode router $A --split test_area --n 200 --bs 16 --out results/ev2_$n.json > logs_ev2_$n.log 2>&1 && echo "ok $n" >> chain2.log || echo "FAILED $n" >> chain2.log
done
echo finished >> chain2.log
