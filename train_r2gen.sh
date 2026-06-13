#!/bin/bash
# R2Gen 训练脚本 (Baseline)
model='r2gen'
max_length=100
epochs=120

export CUDA_VISIBLE_DEVICES=0
python main_train_AllinOne.py \
    --image_dir datas/TCGA-BRCA-feature-uni/pt_files \
    --ann_path datas/WSICAP_REPORT/TCGA-BRCA \
    --split_path datas/WSICAP_REPORT/splits_0.csv \
    --dataset_name wsi_report \
    --model_name $model \
    --token_select uniform_sampling \
    --token_num 256 \
    --max_seq_length $max_length \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --lr_ve 1e-4 \
    --lr_ed 1e-5 \
    --step_size 20 \
    --gamma 0.8 \
    --topk 32 \
    --save_dir results/r2gen_uni \
    --seed 456789 \
    --log_period 1000 \
    --beam_size 3
