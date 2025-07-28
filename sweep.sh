#!/bin/bash

mamba activate reactbench

python ReactBench/main.py config.yaml 

python ReactBench/main.py config.yaml --calc=leftnet 
python ReactBench/main.py config.yaml --calc=leftnet --ckpt_path=/ssd/Code/gad-ff/ckpt/left.ckpt 

python ReactBench/main.py config.yaml --calc=leftnet-d 
python ReactBench/main.py config.yaml --calc=leftnet-d --ckpt_path=/ssd/Code/gad-ff/ckpt/left-df.ckpt 

python ReactBench/main.py config.yaml --calc=equiformer
# --ckpt_path=/ssd/Code/gad-ff/ckpt/eqv2.ckpt 