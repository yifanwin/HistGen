#!/bin/bash
# HistGen 测试脚本
export CUDA_VISIBLE_DEVICES=1
model='histgen'
max_length=100
epochs=120
region_size=96
prototype_num=512

python main_test_AllinOne.py \
    --image_dir datas/TCGA-BRCA-feature-uni/pt_files \
    --ann_path datas/WSICAP_REPORT/TCGA-BRCA \
    --split_path datas/WSICAP_REPORT/splits_0.csv \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 1 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir results/histgen_uni \
    --step_size 1 \
    --gamma 0.8 \
    --seed 42 \
    --log_period 1000 \
    --load results/histgen_uni/model_best.pth \
    --beam_size 3
