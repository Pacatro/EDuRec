#!/bin/bash

datasets=('mars' 'itm' 'doris')

for dataset in "${datasets[@]}"; do
  uv run edurec eval -d "$dataset" -P
done
