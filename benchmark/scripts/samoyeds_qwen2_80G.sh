#!/bin/bash

YAML_LISTS=(
    qwen2_dense_gsm8k
    qwen2_mag_gsm8k
    qwen2_venom_gsm8k
    qwen2_12_16_gsm8k
    qwen2_12_32_gsm8k
)

store_dir=/root/autodl-tmp/output_dir/

for yaml_name in ${YAML_LISTS[@]};do
    echo $yaml_name
    mkdir -p ${store_dir}gsm8k_${yaml_name}

    NM_DISABLE_ANALYTICS=1 CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --standalone --nproc_per_node 1 src/sparseml/transformers/text_generation_fine_tune.py \
        --model_name_or_path Qwen/Qwen2.5-1.5B \
        --dataset_name gsm8k \
        --dataset_config_name main \
        --do_train \
        --do_eval \
        --optim adamw_torch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --preprocessing_num_workers 12 \
        --seed 42 \
        --ddp_find_unused_parameters false \
        --recipe benchmark/recipes/${yaml_name}.yaml \
        --overwrite_output_dir \
        --skip_memory_metrics true \
        --report_to none \
        --output_dir ${store_dir}gsm8k_${yaml_name} \
        | tee ${store_dir}gsm8k_${yaml_name}/execution_log.log
done
