# ReactBench: Benchmarking MLIP Performance in Transition State Search on a Reactive Dataset

ReactBench is an automated transition state(TS) search workflow for benchmarking various MLIPs on a reactive dataset.
Through customizing new calculators or uploading new datasets, ReactBench can be used to evaluate the performance of different MLIPs in TS search.
![ReactBench workflow](./reactbench.jpg)

## Project Structure

```
ReactBench/
├── README.md                    
├── config.yaml                 # Configuration file for running experiments
├── reactbench.jpg              # Workflow diagram image
├── ts1x.tar.gz                 # the whole Transition1x dataset
├── ckpt/                       # Checkpoint directory, files can be downloaded by hugging face.
├── data/                       # Sample reaction coordinate files in XYZ format.
└── ReactBench/                 # Main Python package
    ├── __init__.py            # Package initialization
    ├── main.py                # Main script.
    ├── main_functions.py      
    ├── gsm.py                 
    ├── pysis.py               
    ├── Calculators/           # MLIP calculator implementations
    │   ├── __init__.py        
    │   ├── leftnet.py         # LeftNet calculator.
    │   ├── mace.py            # MACE calculator.    
    │   └── _utils.py          
    ├── utils/                  
    │   ├── __init__.py
    │   ├── parsers.py         
    │   ├── properties.py      
    │   ├── taffi_functions.py 
    │   ├── find_lewis.py      
    │   └── run_pygsm.py       
    └── MLIP/                   
        └── leftnet/           # LeftNet MLIP implementation
        └── mace/              # MACE MLIP implementation
```

## Getting Started

### Installation Guide for Local Machine

```bash
git clone git@github.com:deepprinciple/ReactBench.git
cd ReactBench
mamba create -n reactbench python=3.10
mamba activate reactbench
mamba install -y -c conda-forge openbabel

# install pysisyphus and pygsm
mkdir dependencies 
cd dependencies 
git clone git@github.com:deepprinciple/pysisyphus.git 
cd pysisyphus 
git checkout reactbench 
pip install -e .
cd ..

git clone git@github.com:deepprinciple/pyGSM.git 
cd pyGSM
pip install -e .
cd ../..

cd ReactBench/MLIP/leftnet/ # install leftnet env
pip install -e .
cd ../../..

cd ReactBench/MLIP/mace/ # install mace env
pip install -e .
cd ../../..

pip install -r environment.txt
```

### Installation Guide for SLURM Cluster

Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone git@github.com:BurgerAndreas/ReactBench.git
cd ReactBench
uv venv venv --python 3.11
source venv/bin/activate

module load cuda/12.6
module load gcc/12.3
module load rdkit/2023.09.5 openmm/8.2.0 openbabel/3.1.1 mctc-lib/0.3.1

# install pysisyphus and pygsm
mkdir dependencies 
cd dependencies 
git clone git@github.com:deepprinciple/pysisyphus.git 
cd pysisyphus 
git checkout reactbench 
pip install -e .
cd ..

git clone git@github.com:deepprinciple/pyGSM.git 
cd pyGSM
pip install -e .
cd ../..

cd ReactBench/MLIP/leftnet/ # install leftnet env
pip install -e .
cd ../../..

cd ReactBench/MLIP/mace/ # install mace env
pip install -e .
cd ../../..

uv pip install -r environment.txt
```

I had problems with the compute canada version of wandb, so I installed it manually
```bash
pip uninstall wandb -y

wget https://files.pythonhosted.org/packages/88/c9/41b8bdb493e5eda32b502bc1cc49d539335a92cacaf0ef304d7dae0240aa/wandb-0.20.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -O wandb-0.20.1-py3-none-any.whl

PIP_CONFIG_FILE=/dev/null pip3 install wandb-0.20.1-py3-none-any.whl --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-index
```

Create a .env file in the root directory and set these variables (adjust as needed):
```bash
touch .env
nano .env
```
```bash
# .env
HOMEROOT=${HOME}/ReactBench
# some scratch space where we can write files during training. can be the same as HOMEROOT
PROJECTROOT=${PROJECT}/ReactBench
# the python environment to use (run `which python` to find it)
PYTHONBIN=${HOME}/ReactBench/venv/bin/python
WANDB_ENTITY=...
MPLCONFIGDIR=${PROJECTROOT}/.matplotlib
```


### Setup

1. First, test if the environment is properly set up by running sample data with LEFTNet or MACE calculator:

Download the LeftNet checkpoint from [hugging face](https://huggingface.co/yhong55/ReactBench/tree/main) and place it in the `ckpt` folder.
```bash
mkdir ckpt
curl -L -o ckpt/leftnet-df.ckpt https://huggingface.co/yhong55/ReactBench/resolve/main/leftnet-df.ckpt 
curl -L -o ckpt/leftnet.ckpt https://huggingface.co/yhong55/ReactBench/resolve/main/leftnet.ckpt
```

Download HORM checkpoints with Energy-Force-Hessian Training
```bash
mkdir -p ckpt/horm
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/horm/eqv2.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df.ckpt -O ckpt/horm/left-df.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left.ckpt -O ckpt/horm/left.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha.ckpt -O ckpt/horm/alpha.ckpt
```

Download Transition1x validation subset recomputed, 960 datapoints
```bash
mkdir -p data 
tar -xzf ts1x.tar.gz -C data
find data/ts1x -type f | wc -l # 960
```

### Run

```bash
python ReactBench/main.py config.yaml


python ReactBench/main.py config.yaml --calc=leftnet 
python ReactBench/main.py config.yaml --calc=leftnet --ckpt_path=ckpt/horm/left.ckpt 

python ReactBench/main.py config.yaml --calc=leftnet-d 
python ReactBench/main.py config.yaml --calc=leftnet-d --ckpt_path=ckpt/horm/left-df.ckpt 

python ReactBench/main.py config.yaml --calc=equiformer
--ckpt_path=ckpt/horm/eqv2.ckpt 
``` 

calc can be: leftnet, leftnet-d, mace-pretrain, mace-finetuned


2. To test other MLIPs, you can create your own calculator by following these steps:

   a. Go to `ReactBench/Calculators` directory

   b. Use `leftnet` calculator as a reference implementation
   
   c. Create a new calculator class for your MLIP of interest
   
   d. Update `__init__.py` to register your new calculator
   
   e. Modify the `calc` parameter in `config.yaml` to use your calculator


### Citation

```
@article{https://doi.org/10.1002/advs.202506240,
        author = {Zhao, Qiyuan and Han, Yunhong and Zhang, Duo and Wang, Jiaxu and Zhong, Peichen and Cui, Taoyong and Yin, Bangchen and Cao, Yirui and Jia, Haojun and Duan, Chenru},
        title = {Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models},
        journal = {Advanced Science},
        pages = {e06240},
        doi = {https://doi.org/10.1002/advs.202506240}
}
```


### License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).