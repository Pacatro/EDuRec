#!/bin/bash

GPU_ID="$1"

datasets=('mars' 'itm' 'doris')

for dataset in "${datasets[@]}"; do
  if [[ -v GPU_ID ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" uv run edurec train -d "$dataset" -P -S
  else
    uv run edurec train -d "$dataset" -P -S
  fi
done
