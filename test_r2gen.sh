#!/bin/bash
# R2Gen 测试脚本 (Baseline)
model='r2gen'
max_length=100
epochs=30

python main_test_AllinOne.py \
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
    --step_size 1 \
    --topk 32 \
    --save_dir results/r2gen_uni_test \
    --step_size 1 \
    --gamma 0.8 \
    --seed 42 \
    --log_period 1000 \
    --load results/r2gen_uni/model_best.pth \
    --beam_size 3
