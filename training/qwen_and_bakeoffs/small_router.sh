#!/bin/bash
# router-mode comparison: same 10k examples, answer-only loss, then router-mode eval on unseen neighbourhoods
cd "$(dirname "$0")"; source .venv-unsloth/bin/activate; mkdir -p results
for spec in "Qwen3.5-0.8B 16 1" "Qwen3.5-2B 16 1" "Qwen3.5-4B 16 1" "Qwen3.5-9B 8 2"; do
  set -- $spec; n=$1; bs=$2; acc=$3
  echo "=== $n $(date +%T)" >> small_router.log
  python -u train_unsloth.py --model unsloth/$n --out runs/rt_$n --n 10000 --epochs 1 --bs $bs --accum $acc --answer_only > logs_rt_$n.train 2>&1 \
   && python -u eval_reason.py --mode router --adapter runs/rt_$n --split test_area --n 200 --bs 16 --out results/rt_${n}_area.json > logs_rt_$n.eval 2>&1 \
   && echo "ok $n" >> small_router.log || echo "FAILED $n" >> small_router.log
done
echo finished >> small_router.log
