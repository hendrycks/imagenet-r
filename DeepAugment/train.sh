#!/bin/bash

source ~/pytorch.sh

python train.py --dist-url 'tcp://127.0.0.1:32767' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /data/imagenet
