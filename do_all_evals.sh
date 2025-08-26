
cd ../gad-ff
source .venv/bin/activate

# ~3h
cd ../gad-ff
python scripts/eval_horm.py --redo=True
python scripts/eval_horm.py --ckpt_path=/ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt --hessian_method=predict --redo=True

# ~24h
# has to be before compare_hessian_to_dft.py
cd ../ReactBench
python ReactBench/main.py config.yaml --calc=equiformer --hessian_method=autograd --redo_all=True
python ReactBench/main.py config.yaml --calc=equiformer --ckpt_path=alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956 --hessian_method=predict --redo_all=True

# ~3h
cd ../ReactBench
uv run compare_hessian_to_dft.py equiformer_alldatagputwoalphadrop0droppathrate0projdrop0 --which final --redo True
uv run compare_hessian_to_dft.py equiformer_alldatagputwoalphadrop0droppathrate0projdrop0 --which initial --redo True

cd ../gad-ff
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --thresh gau --redo True;
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --thresh gau_loose --redo True;
uv run scripts/second_order_relaxation_pysiyphus.py --max_samples 30 --thresh gau_loose --coord cart --redo True;
