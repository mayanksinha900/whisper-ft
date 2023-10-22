#!/bin/bash

python main.py -t $1
model=$(cat tests/tmp.txt)
echo "Model: $model"
ct2-transformers-converter --model "models/$model/final_model" --output_dir "models/$model/ct2_optimized" --quantization $2
cp "models/$model/final_model/tokenizer.json" "models/$model/ct2_optimized/tokenizer.json" 