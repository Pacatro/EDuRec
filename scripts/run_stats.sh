#!/bin/bash

topks=(5 10 15)

for topk in "${topks[@]}"; do
  uv run edurec stats --top_k "$topk"
  uv run edurec stats --top_k "$topk"
done
