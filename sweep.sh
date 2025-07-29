#!/bin/bash

mamba activate reactbench

# test runs
# python ReactBench/main.py config.yaml --inp_path=data
# python ReactBench/main.py config.yaml --calc=leftnet --inp_path=data # H ML TS IP checkpoint
# python ReactBench/main.py config.yaml --calc=leftnet --ckpt_path=/ssd/Code/gad-ff/ckpt/left.ckpt --inp_path=data # HORM checkpoint
# python ReactBench/main.py config.yaml --calc=leftnet-d --inp_path=data # H ML TS IP checkpoint
# python ReactBench/main.py config.yaml --calc=leftnet-d --ckpt_path=/ssd/Code/gad-ff/ckpt/left-df.ckpt --inp_path=data # HORM checkpoint
# python ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=/ssd/Code/gad-ff/ckpt/horm/eqv2.ckpt --inp_path=data # HORM checkpoint


python ReactBench/main.py config.yaml 
python ReactBench/main.py config.yaml --calc=leftnet # H ML TS IP checkpoint
python ReactBench/main.py config.yaml --calc=leftnet --ckpt_path=/ssd/Code/gad-ff/ckpt/left.ckpt # HORM checkpoint
# python ReactBench/main.py config.yaml --calc=leftnet-d # H ML TS IP checkpoint
# python ReactBench/main.py config.yaml --calc=leftnet-d --ckpt_path=/ssd/Code/gad-ff/ckpt/left-df.ckpt # HORM checkpoint
python ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=/ssd/Code/gad-ff/ckpt/horm/eqv2.ckpt # HORM checkpoint