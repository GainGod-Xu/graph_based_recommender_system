#!/bin/bash
# run benchmark dataset
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python models/CompGCN/run.py 
