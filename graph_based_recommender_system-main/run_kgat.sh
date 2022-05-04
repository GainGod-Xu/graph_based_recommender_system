#!/bin/bash

# python3 -m pip install torch==1.6.0
# python3 -m pip install numpy==1.21.4
# python3 -m pip install pandas==1.3.5
# python3 -m pip install scipy==1.5.2
# python3 -m pip install tqdm==4.62.3
# python3 -m pip install sklearn

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
#python models/KGAT-pytorch-master/main_kgat.py --data_name last-fm --use_pretrain 0 --use_focal 0 --lr 0.001
#python models/KGAT-pytorch-master/main_kgat.py --data_name last-fm --use_pretrain 0 --use_focal 0.9999 --lr 0.001
# python models/KGAT-pytorch-master/main_kgat.py --data_name amazon-review-kgat_data --use_pretrain 0 --use_focal 0 --lr 0.0001 --embed_dim 200
# python models/KGAT-pytorch-master/main_kgat.py --data_name amazon-review-kgat_data --use_pretrain 0 --use_focal 0.9999 --lr 0.0001 --embed_dim 200
#python models/KGAT-pytorch-master/main_kgat.py --data_name amazon-product-review/kgat_data --use_pretrain 0 --use_focal 0 --lr 0.0001 --embed_dim 200 --stopping_steps 30
#python models/KGAT-pytorch-master/main_kgat.py --data_name amazon-product-review/kgat_data --use_pretrain 0 --use_focal 0.9999 --lr 0.0001 --embed_dim 200 --stopping_steps 30
python models/KGAT-pytorch-master/main_kgat.py


