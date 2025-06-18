#!/usr/bin/env bash
PORT=$((RANDOM + 10000))
GPUS=$1
CONFIG=$2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG ${@:3}
