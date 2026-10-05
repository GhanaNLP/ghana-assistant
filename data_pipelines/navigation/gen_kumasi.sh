#!/bin/bash
cd "$(dirname "$0")"; WORKERS=96 python3 -u generate_queries.py kumasi >> gen_kumasi.log 2>&1; echo done > gen_kumasi.done
