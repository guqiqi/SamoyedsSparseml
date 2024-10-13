#!/bin/bash

YAML_LISTS=(
    # tiny_llama_12_16
    # tiny_llama_12_16_layer_first
    # tiny_llama_12_32
    # tiny_llama_12_32_0rate
    # tiny_llama_12_32_layer_first
    # tiny_llama_12_64
)

for yaml_name in ${YAML_LISTS[@]};do
    echo $yaml_name
    mkdir -p benchmark/output_dir/gsm8k_${yaml_name}

    # NM_DISABLE_ANALYTICS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --standalone --nproc_per_node 2 src/sparseml/transformers/text_generation_fine_tune.py \
    #     --model_name_or_path TinyLlama/TinyLlama_v1.1 \
    #     --dataset_name open_platypus \
    #     --bf16 \
    #     --do_train \
    #     --do_eval \
    #     --optim adamw_torch \
    #     --evaluation_strategy epoch \
    #     --save_strategy epoch \
    #     --save_total_limit 3 \
    #     --per_device_train_batch_size 8 \
    #     --per_device_eval_batch_size 8 \
    #     --max_seq_length 384 \
    #     --doc_stride 128 \
    #     --preprocessing_num_workers 32 \
    #     --seed 42 \
    #     --ddp_find_unused_parameters false \
    #     --recipe benchmark/recipes/${yaml_name}.yaml \
    #     --overwrite_output_dir \
    #     --skip_memory_metrics true \
    #     --report_to none \
    #     --output_dir benchmark/output_dir/${yaml_name} \
    #     --distill_teacher benchmark/output_dir/tiny_llama_teacher \
    #     --gradient_accumulation_steps 4 \
    #     | tee benchmark/output_dir/${yaml_name}/execution_log.log
done 

YAML_LISTS=(
    # tiny_llama_12_16_gsm8k
    tiny_llama_12_32_gsm8k
    # tiny_llama_12_16_layer_first
    # tiny_llama_12_32
    # tiny_llama_12_32_0rate
    # tiny_llama_12_32_layer_first
    # tiny_llama_12_64
)

for yaml_name in ${YAML_LISTS[@]};do
    echo $yaml_name
    mkdir -p benchmark/output_dir/gsm8k_${yaml_name}

    NM_DISABLE_ANALYTICS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --standalone --nproc_per_node 2 src/sparseml/transformers/text_generation_fine_tune.py \
        --model_name_or_path TinyLlama/TinyLlama_v1.1 \
        --dataset_name gsm8k \
        --dataset_config_name main \
        --bf16 \
        --do_train \
        --do_eval \
        --optim adamw_torch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --preprocessing_num_workers 32 \
        --seed 42 \
        --ddp_find_unused_parameters false \
        --recipe benchmark/recipes/${yaml_name}.yaml \
        --overwrite_output_dir \
        --skip_memory_metrics true \
        --report_to none \
        --output_dir benchmark/output_dir/gsm8k_${yaml_name} \
        | tee benchmark/output_dir/gsm8k_${yaml_name}/execution_log.log
done
