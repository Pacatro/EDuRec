#!/bin/bash

topks=(5 10 15)

for topk in "${topks[@]}"; do
  uv run src/main.py -st --top_k "$topk"
  uv run src/main.py -st --top_k "$topk"
done
