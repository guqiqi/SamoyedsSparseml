#!/bin/bash

YAML_LISTS=(
    tiny_llama_12_64
)

for yaml_name in ${YAML_LISTS[@]};do
    echo $yaml_name

    # echo "NM_DISABLE_ANALYTICS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --standalone --nproc_per_node 2 src/sparseml/transformers/text_generation_fine_tune.py \
    #     --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    #     --dataset_name open_platypus \
    #     --do_train \
    #     --bf16 \
    #     --do_eval \
    #     --optim adamw_torch \
    #     --evaluation_strategy epoch \
    #     --save_strategy epoch \
    #     --save_total_limit 3 \
    #     --per_device_train_batch_size 8 \
    #     --per_device_eval_batch_size 8 \
    #     --learning_rate 8e-5 \
    #     --max_seq_length 384 \
    #     --doc_stride 128 \
    #     --preprocessing_num_workers 1 \
    #     --seed 42 \
    #     --num_train_epochs 50 \
    #     --recipe benchmark/recipes/${yaml_name}.yaml \
    #     --overwrite_output_dir \
    #     --skip_memory_metrics true \
    #     --report_to none \
    #     --output_dir benchmark/output_dir/${yaml_name} \  "

    # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python src/sparseml/transformers/text_generation_fine_tune.py \
    NM_DISABLE_ANALYTICS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --standalone --nproc_per_node 2 src/sparseml/transformers/text_generation_fine_tune.py \
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset_name open_platypus \
        --do_train \
        --bf16 \
        --do_eval \
        --optim adamw_torch \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --learning_rate 8e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --preprocessing_num_workers 1 \
        --seed 42 \
        --num_train_epochs 50 \
        --recipe benchmark/recipes/${yaml_name}.yaml \
        --overwrite_output_dir \
        --skip_memory_metrics true \
        --report_to none \
        --output_dir benchmark/output_dir/${yaml_name}   
done

        # --distill_teacher ./benchmark/output_dir/tiny_llama_teacher \
