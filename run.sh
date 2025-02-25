#!/bin/bash

current_time=$(date "+%Y%m%d-%H%M%S")
log_file="logs/run_${current_time}.log"

mkdir -p logs


python3 run.py \
  --data_name "LastFM" \
  --data_dir "./data/" \
  --use_pretrain 0 \
  --cf_batch_size 2048 \
  --kg_batch_size 1024 \
  --test_batch_size 8912 \
  --laplacian_type "symmetric" \
  --aggregation_type "gcn" \
  --conv_dim_list "[128, 128, 64]" \
  --mess_dropout "[0.1, 0.1, 0.1]" \
  --kg_l2loss_lambda 1e-5 \
  --cf_l2loss_lambda 1e-5 \
  --lr 0.0001 \
  --n_epoch 10 \
  --stopping_steps 20 \
  --cf_print_every 10 \
  --dropout 0.01 \
  --evaluate_every 5 \
  --Ks "[10, 20]" \
  --save_dir "./models/kgbert_model_ngcf2" \
  > "$log_file" 2>&1

echo "Training complete. Logs saved to $log_file"
