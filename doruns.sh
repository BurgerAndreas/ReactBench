
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


# ~24h
# has to be before compare_hessian_to_dft.py, since it saves Hessians and geometry xyz files
# cd ../ReactBench
# uv run ReactBench/main.py config.yaml --calc=equiformer --hessian_method=autograd --config_path=null # --redo_all=True
# uv run ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=$HPCKPT --hessian_method=predict 


cd ../gad-ff
# --redo True
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --xyz t1x --thresh gau --max_cycles 100
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --xyz t1x --thresh gau --max_cycles 100 --coord cart

# Optional: equivariance test