#!/bin/bash

export YAML_NAME=obs_216v128_gradual_pair
export RECIPE=benchmark/recipes/${YAML_NAME}.yaml

#export RECIPE=research/optimal_BERT_surgeon_oBERT/recipes/30epochs_init30_4block80_squad.yaml

# export HF_HUB_OFFLINE=1

#uncomment to run on a single-gpu
#CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/question_answering.py \

# python -m torch.distributed.launch --nproc_per_node=1 src/sparseml/transformers/question_answering.py \
# CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/question_answering.py \
# torchrun --nnodes 1 --standalone --nproc_per_node 8 src/sparseml/transformers/question_answering.py \

# YAML_LISTS=(
#     obs_samoyeds_gradual_pair_12_4 obs_samoyeds_gradual_pair_12_16 obs_samoyeds_gradual_pair_12_64 obs_samoyeds_gradual_pair_12_128 obs_samoyeds_gradual_pair_48_4 obs_samoyeds_gradual_pair_48_16 obs_samoyeds_gradual_pair_48_64 obs_samoyeds_gradual_pair_48_128
# )

YAML_LISTS=(
    obs_samoyeds_gradual_pair_12_4
    obs_samoyeds_gradual_pair_12_16
    obs_samoyeds_gradual_pair_12_64 obs_samoyeds_gradual_pair_12_128 obs_samoyeds_gradual_pair_48_4 obs_samoyeds_gradual_pair_48_16 obs_samoyeds_gradual_pair_48_64 obs_samoyeds_gradual_pair_48_128 obs_samoyeds_gradual_pair_816_4 obs_samoyeds_gradual_pair_816_16 obs_samoyeds_gradual_pair_816_64 obs_samoyeds_gradual_pair_816_128
)

for yaml_name in ${YAML_LISTS[@]};do
    echo $yaml_name

    export YAML_NAME=$yaml_name
    export RECIPE=benchmark/recipes/${YAML_NAME}.yaml
        
    torchrun --nnodes 1 --standalone --nproc_per_node 8 src/sparseml/transformers/question_answering.py \
    --distill_teacher neuralmagic/oBERT-teacher-squadv1 \
    --model_name_or_path bert-base-uncased \
    --dataset_name rajpurkar/squad \
    --do_train \
    --fp16 \
    --do_eval \
    --optim adamw_torch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 8e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --preprocessing_num_workers 16 \
    --seed 42 \
    --num_train_epochs 50 \
    --recipe ${RECIPE} \
    --overwrite_output_dir \
    --skip_memory_metrics true \
    --report_to none \
    --output_dir benchmark/output_dir/${YAML_NAME} #\
    #--max_train_samples 1024 \
    #--max_eval_samples 1024 \
    #--max_predict_samples 1024

done;
