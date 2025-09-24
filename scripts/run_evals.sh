#!/bin/bash

datasets=('mars' 'itm' 'coursera')
topks=(5 10 15)

for dataset in "${datasets[@]}"; do
  for topk in "${topks[@]}"; do
    echo "Evaluating $dataset dataset on the Top-$topk recommendations"
    uv run src/main.py -e -ds "$dataset" --top_k "$topk"
    uv run src/main.py -s -ds "$dataset" --top_k "$topk"
  done
done
