#!/bin/bash
cd "$(dirname "$0")"; while [ ! -f expand.done ]; do sleep 10; done
WORKERS=96 python3 -u generate_queries.py accra >> gen_accra.log 2>&1; echo done > gen_accra.done
