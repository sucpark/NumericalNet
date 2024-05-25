#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

DATASET_PATH="./datasets/numerical_dataset_v1_size50000.xlsx"
TEST_NAME="test_v1.2"
NUM_EPOCHS=50
BATCH_SIZE=512
LR=0.0005
EARLY_STOPPING_PATIENCE=5

# Run Python training script with arguments
python train.py --dataset_path $DATASET_PATH \
                --experiment_name $TEST_NAME \
                --num_epochs $NUM_EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --early_stopping_patience $EARLY_STOPPING_PATIENCE