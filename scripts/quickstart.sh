#!/bin/bash

python data/prepare_dota.py
python data/build_fewshot_split.py
python tools/train_base.py --config configs/train_base.yaml
for k in 1 2 5 10; do
  python tools/finetune_fewshot.py --config configs/finetune_k${k}.yaml --shots $k
  python tools/evaluate.py --model_path checkpoints/finetune_k${k}/final.pth --split test
done