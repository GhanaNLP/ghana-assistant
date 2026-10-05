#!/bin/bash
cd "$(dirname "$0")"; source .venv/bin/activate; mkdir -p results
for spec in "s_falcon100m|tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct" "s_smol135|HuggingFaceTB/SmolLM2-135M-Instruct" "s_flant5small|google/flan-t5-small" "s_flant5base|google/flan-t5-base"; do
  n=${spec%%|*}; m=${spec##*|}; echo "=== $n $(date +%T)" >> small_bakeoff.log
  python -u train_small.py --model $m --out runs/$n --n 50000 > logs_$n.train 2>&1 \
   && python -u eval_small.py --run runs/$n --out results/$n.json > logs_$n.eval 2>&1 \
   && echo "ok $n" >> small_bakeoff.log || echo "FAILED $n" >> small_bakeoff.log
done
echo finished >> small_bakeoff.log
