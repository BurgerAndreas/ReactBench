import hydra
import re
import shutil
import yaml
from omegaconf import DictConfig
import os
import torch
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


@hydra.main(
    version_base=None, config_path="../gad-ff/configs", config_name="train_eigen"
)
def main(cfg: DictConfig) -> None:
    # get extra args from config
    hessian_method = cfg.eval_hessian_method
    max_samples = cfg.get("eval_max_samples", 1000)

    ckpt_path = get_ckpt_name(cfg)
    if ckpt_path is None:
        print("No checkpoint found, exiting")
        return

    print(f"Evaluating checkpoint: {ckpt_path}")

    REACT_DIR = "/project/aip-aspuru/aburger/ReactBench"
    GAD_DIR = "/project/aip-aspuru/aburger/gad-ff"
    if not os.path.exists(GAD_DIR):
        REACT_DIR = "/ssd/Code/ReactBench"
        GAD_DIR = "/ssd/Code/gad-ff"

    # copy the checkpoint to the evaluation directory
    # so we save it before it might be overwritten by the next checkpoint
    CHECKPOINT_DIR = f"{GAD_DIR}/ckpteval"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    shutil.copy(ckpt_path, CHECKPOINT_DIR)

    # load the checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    # model_type = model_config["model_type"]
    print(f"Model name: {model_name}")

    # This will launch a wandb run
    # TODO: init wandb run here, pass wandb_run_id to evaluate
    hormmetrics = evaluate(
        lmdb_path="ts1x-val.lmdb",
        checkpoint_path=ckpt_path,
        config_path="auto",
        hessian_method=hessian_method,
        max_samples=max_samples,
        wandb_run_id=None,
        wandb_kwargs={"tags": ["evalckpt"]},
    )

    #######################################################################
    # Transition state search workflow
    #######################################################################
    print("\n" + "-" * 80)
    print("Transition state search workflow")
    print("-" * 80)

    parameters_yaml = f"{REACT_DIR}/config_killarney.yaml"
    parameters = yaml.load(open(parameters_yaml, "r"), Loader=yaml.FullLoader)

    parameters["ckpt_path"] = ckpt_path
    parameters["config_path"] = "auto"
    parameters["hessian_method"] = hessian_method
    parameters["reactbench_path"] = REACT_DIR
    # TODO: also support leftnet, leftnet-df, mace, alphanet
    parameters["calc"] = "equiformer"

    # This will launch another wandb run
    launch_tssearch_processes(
        parameters, wandb_run_id=None, wandb_kwargs={"tags": ["evalckpt"]}
    )

    return


if __name__ == "__main__":
    """
    Use the same args as in train_eigen.py
    but add:
    +eval_hessian_method=autograd/predict
    +eval_max_samples=1000
    """
    main()
