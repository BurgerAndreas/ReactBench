import hydra
import re
import shutil
import yaml
from omegaconf import DictConfig
import os
from gadff.logging_utils import name_from_config, find_latest_checkpoint
from gadff.horm.eval_horm import evaluate 
from ReactBench.main import launch_tssearch_processes


def get_ckpt_name(cfg: DictConfig):
    ###########################################
    # Trainer checkpoint loading
    ###########################################
    # get checkpoint name
    run_name_ckpt = name_from_config(cfg, is_checkpoint_name=True)
    checkpoint_name = re.sub(r"[^a-zA-Z0-9]", "", run_name_ckpt)
    if len(checkpoint_name) <= 1:
        checkpoint_name = "base"
    print(f"Checkpoint name: {checkpoint_name}")

    if cfg.ckpt_trainer_path is None:
        # Auto-resume logic: find existing trainer checkpoint with same base name
        latest_ckpt = find_latest_checkpoint(checkpoint_name, cfg.project)
        if latest_ckpt:
            cfg.ckpt_trainer_path = latest_ckpt
            print(f"Auto-resume: Will resume from {latest_ckpt}")
        else:
            print("Auto-resume: No existing checkpoints found, starting fresh")

    print(f"\n{cfg.ckpt_trainer_path}")

    return cfg.ckpt_trainer_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_eigen")
def main(cfg: DictConfig) -> None:
    ckpt_path = get_ckpt_name(cfg)
    
    REACT_DIR = "/project/aip-aspuru/aburger/ReactBench"
    GAD_DIR = "/project/aip-aspuru/aburger/gad-ff"
    
    if not os.path.exists(GAD_DIR):
        REACT_DIR = "/ssd/Code/ReactBench"
        GAD_DIR = "/ssd/Code/gad-ff"

    CHECKPOINT_DIR = f"{GAD_DIR}/ckpteval"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    shutil.copy(ckpt_path, CHECKPOINT_DIR)

    
    evaluate(
        lmdb_path=f"{GAD_DIR}/datasets/ts1x-val.lmdb",
        checkpoint_path=ckpt_path,
        config_path="auto",
        hessian_method="predict",
        max_samples=None,
    )

    parameters_yaml = f"{REACT_DIR}/config.yaml"
    parameters = yaml.load(open(parameters_yaml, "r"), Loader=yaml.FullLoader)
    
    parameters["ckpt_path"] = ckpt_path
    parameters["config_path"] = "auto"
    parameters["hessian_method"] = "predict"
    parameters["reactbench_path"] = REACT_DIR
    parameters["calc"] = "equiformer"
    
    launch_tssearch_processes(parameters)
    
    return

if __name__ == "__main__":
    main()
