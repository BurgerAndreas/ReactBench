
# findckpt 581483
# l=3 luca8 + luca/10 bz=128
# /project/aip-aspuru/aburger/gad-ff/checkpoint/gadff/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746

# getreactckpt
# rsync -avz --progress aburger@killarney.alliancecan.ca:/project/aip-aspuru/aburger/gad-ff/checkpoint/gadff/$1/last.ckpt /ssd/Code/ReactBench/ckpt/hesspred/$1.ckpt

# /ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt
export HPCKPT="/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt"


cd ../gad-ff
source .venv/bin/activate

# MAE, cosine similarity, ...
# ~3h
cd ../gad-ff
# other HORM models
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/alpha.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/left-df.ckpt --redo=True; 
uv run scripts/eval_horm.py --max_samples=1000 --ckpt_path=ckpt/eqv2.ckpt --redo=True; 
uv run scripts/eval_horm.py --ckpt_path=$HPCKPT --hessian_method=predict  --max_samples=1000;

# Plot results in ../gad-ff/results/eqv2_ts1x-val_autograd_metrics.csv / wandb export
uv run scripts/plot_frequency_analysis.py

# Speed and memory comparison (plot included)
# ~1h
cd ../gad-ff
uv run scripts/speed_comparison.py speed --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
uv run scripts/speed_comparison.py speed --dataset ts1x-val.lmdb --max_samples_per_n 100

# ~24h
# has to be before compare_hessian_to_dft.py, since it saves Hessians and geometry xyz files
cd ../ReactBench
uv run ReactBench/main.py config.yaml --calc=equiformer --hessian_method=autograd --redo_all=True --config_path=null
uv run ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=$HPCKPT --hessian_method=predict --redo_all=True

# plot

# ~3h
cd ../ReactBench
uv run compare_hessian_to_dft.py equiformer_alldatagputwoalphadrop0droppathrate0projdrop0 --which final --redo True
uv run compare_hessian_to_dft.py equiformer_alldatagputwoalphadrop0droppathrate0projdrop0 --which initial --redo True

cd ../gad-ff
# --redo True
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30--xyz t1x --thresh gau
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30--xyz t1x --thresh gau_loose
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30--xyz t1x --thresh gau_loose --coord cart
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 10 --xyz t1x --max_cycles 15
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 10 --xyz t1x --max_cycles 15 --coord cart

# Optional: equivariance test