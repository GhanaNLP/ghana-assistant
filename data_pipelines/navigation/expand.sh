#!/bin/bash
cd "$(dirname "$0")"
for c in kumasi accra; do python3 -u build_scenarios.py $c 32000 > build32k_$c.log 2>&1; done
echo built > expand.done
