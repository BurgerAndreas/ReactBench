
# source /ssd/Code/ReactBench/doruns.sh

# /ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt
export HPCKPT="/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt"


cd ../gad-ff
source .venv/bin/activate

# cd ../gad-ff
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/alpha.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left-df.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/eqv2.ckpt --redo=True; 
uv run scripts/eval_horm.py --ckpt_path=$HPCKPT --hessian_method=predict --max_samples=1000 --redo=True;


# ~24h
# has to be before compare_hessian_to_dft.py, since it saves Hessians and geometry xyz files
# cd ../ReactBench
# uv run ReactBench/main.py config.yaml --calc=equiformer --hessian_method=autograd --config_path=null --only_cnt_results=True # --redo_all=True
# uv run ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=$HPCKPT --hessian_method=predict --only_cnt_results=True 
# uv run ReactBench/main.py config.yaml --calc=leftnet --hessian_method=autograd --config_path=null --ckpt_path=/ssd/Code/ReactBench/ckpt/horm/left.ckpt --only_cnt_results=True  #--redo_all=True
uv run ReactBench/main.py config.yaml --calc=leftnet-d --hessian_method=autograd --config_path=null --ckpt_path=/ssd/Code/ReactBench/ckpt/horm/left-df.ckpt --only_cnt_results=True  #--redo_all=True
# # uv run ReactBench/main.py config.yaml --calc=mace-finetuned --hessian_method=autograd 



# uv run scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
# uv run scripts/speed_comparison_incltransform.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt


# cd ../gad-ff
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --xyz t1x --thresh gau --max_cycles 150
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --thresh gau --max_cycles 150 --xyz /ssd/Code/gad-ff/data/t1x_val_reactant_hessian_100_tight2_noiserms0.03.h5
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 10 --thresh gau --max_cycles 150 --xyz /ssd/Code/gad-ff/data/t1x_val_reactant_hessian_100_tight2_noiserms0.03.h5
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 80 --thresh gau --max_cycles 150 --xyz /ssd/Code/gad-ff/data/t1x_val_reactant_hessian_100_tight2_noiserms0.03.h5

# # /ssd/Code/gad-ff/data/t1x_val_reactant_hessian_100_noiserms0.03.h5
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --xyz t1x_0.03 --thresh gau --max_cycles 150 
# # /ssd/Code/gad-ff/data/t1x_val_reactant_hessian_100_noiserms0.05.h5
# uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --xyz t1x_0.05 --thresh gau --max_cycles 150

# Optional: equivariance test