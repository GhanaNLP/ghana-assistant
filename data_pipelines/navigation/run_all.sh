#!/bin/bash
cd "$(dirname "$0")"
for c in kumasi accra; do
  [ -f ${c}_scenarios.jsonl ] || python3 -u build_scenarios.py $c 4000 > build_$c.log 2>&1
  python3 -u generate_queries.py $c >> gen_$c.log 2>&1
done
echo done > run_all.done
