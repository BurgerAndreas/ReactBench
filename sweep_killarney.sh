#!/bin/bash

source venv/bin/activate
module load cuda/12.6
module load gcc/12.3
module load rdkit/2023.09.5 openmm/8.2.0 openbabel/3.1.1 mctc-lib/0.3.1


sbatch killarney.sh ReactBench/main.py config.yaml 

sbatch killarney.sh ReactBench/main.py config.yaml --calc=leftnet 
sbatch killarney.sh ReactBench/main.py config.yaml --calc=leftnet --ckpt_path=/project/aip-aspuru/aburger/ReactBench/ckpt/horm/left.ckpt 

sbatch killarney.sh ReactBench/main.py config.yaml --calc=leftnet-d 
sbatch killarney.sh ReactBench/main.py config.yaml --calc=leftnet-d --ckpt_path=/project/aip-aspuru/aburger/ReactBench/ckpt/horm/left-df.ckpt 

sbatch killarney.sh ReactBench/main.py config.yaml --calc=equiformer 

# --ckpt_path=/project/aip-aspuru/aburger/ReactBench/ckpt/horm/eqv2.ckpt 