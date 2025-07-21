from typing import List, Optional, Tuple
from collections import OrderedDict
from uuid import uuid4
import torch

# from pl_trainer import DDPMModule
from potential_module import PotentialModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet


model_type = "leftnet"
version = "4L40-df-cont"
project = "test"
# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=True,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
)


if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=1e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 200
T_mult = 2
training_config = dict(
    datadir="/deepprinciple-proj-dev/datasets/transition-1x/",
    bz=32,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=10,
    ),  # step
)


node_nfs: List[int] = [9] * 1  # 3 (pos) + 5 (cat) + 1 (charge)
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["structure"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = False
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
enforce_same_encoding = None
use_autograd = False
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]
timesteps = 5000

use_pretrain = False
source = None

# if use_pretrain:
    # checkpoint_path = "/home/ubuntu/efs/OA_ReactDiff/oa_reactdiff/trainer/checkpoint/TS1x-TSDiff/leftnet-0-0f522f4c30fa/ddpm-epoch=719-val-error_t_0=0.280.ckpt"  # has time
    # checkpoint_path = "/home/ubuntu/efs/OA_ReactDiff/oa_reactdiff/trainer/checkpoint/TS1x-TSDiff/leftnet-notime-766403265642/ddpm-epoch=219-val-error_t_0=0.293.ckpt"  # no time
    # ddpm_trainer = DDPMModule.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path,
    #     map_location="cpu",
    # )
    # source = {
    #     "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
    #     "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
    #     "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
    # }
    
    # new_source = {
    #     "encoders": OrderedDict(),
    #     "decoders": OrderedDict(),
    # }
    # for k, v in source["encoders"].items():
    #     if k[0] == "0":
    #         new_source["encoders"][k] = v
    # for k, v in source["decoders"].items():
    #     if k[0] == "0":
    #         new_source["decoders"][k] = v
    # source["encoders"] = new_source["encoders"]
    # source["decoders"] = new_source["decoders"]

seed_everything(1, workers=True)
ddpm = PotentialModule(
    model_config=model_config,
    optimizer_config=optimizer_config,
    training_config=training_config,
    node_nfs=node_nfs,
    edge_nf=edge_nf,
    condition_nf=condition_nf,
    fragment_names=fragment_names,
    pos_dim=pos_dim,
    edge_cutoff=edge_cutoff,
    model=model,
    enforce_same_encoding=enforce_same_encoding,
    use_autograd=use_autograd,
    source=source,
    timesteps=timesteps,
    condition_time=condition_time,
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
        entity="deep-principle",
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(
            ddpm.confidence, log="all", log_freq=100, log_graph=False
        )
    except:
        pass

ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ff-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}",
    every_n_epochs=10,
    save_top_k=-1,
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback())

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy ='auto'
print(strategy, devices)
trainer = Trainer(
    max_epochs=10000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=50,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=400,
    limit_val_batches=20,
    # resume_from_checkpoint=checkpoint_path,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)

# print("Finished training, continue training for 3000 epochs.")
# trainer.fit_loop.max_epochs += 3000
# trainer.fit(ddpm) 
