#!/bin/bash

{
  NM_DISABLE_ANALYTICS=1 CUDA_VISIBLE_DEVICES=0,1 python src/sparseml/transformers/text_generation_fine_tune.py \
    --output_dir benchmark/output_dir/tiny_llama_teacher \
    --recipe benchmark/recipes/tiny_llama_teacher.yaml \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name open_platypus \
    --bf16 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --seed 42
} && mv benchmark/output_dir/tiny_llama_teacher/recipe.yaml benchmark/output_dir/tiny_llama_teacher/recipe.yaml.back