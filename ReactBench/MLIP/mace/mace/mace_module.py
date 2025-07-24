from typing import Dict, List, Optional, Tuple

from pathlib import Path
import torch
from torch import nn

from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from pytorch_lightning import LightningModule
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    CosineSimilarity,
)

from mace import data
from mace.tools.utils import AtomicNumberTable
from mace.data.atomic_data import get_data_loader
from mace.calculators import mace_off
from mace.utils import average_over_batch_metrics, pretty_print
import mace.utils as utils_diff


LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        super().__init__()

        self.potential = mace_off(model=model_config["pretrained"]).models[0]

        for param in self.potential.parameters():
            param.requires_grad = True

        # ---MACE-OFF Specific Configurations---
        self.mace_off_atom_reference = {
            35: -70045.28385080204,
            6: -1030.5671648271828,
            17: -12522.649269035726,
            9: -2715.318528602957,
            1: -13.571964772646918,
            53: -8102.524593409054,
            7: -1486.3750255780376,
            8: -2043.933693071156,
            15: -9287.407133426237,
            16: -10834.4844708122,
        }
        self.r_max = 5.0
        self.z_table = AtomicNumberTable(
            sorted(list(self.mace_off_atom_reference.keys()))
        )

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = utils_diff.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.MSELoss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.potential.parameters(), **self.optimizer_config
        )
        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = data.dataset_from_sharded_hdf5(
                self.training_config["datadir"] + "/train/",
                r_max=self.r_max,
                z_table=self.z_table,
                heads=None,
                head=None,
            )
            self.val_dataset = data.dataset_from_sharded_hdf5(
                self.training_config["datadir"] + "/val/",
                r_max=self.r_max,
                z_table=self.z_table,
                heads=None,
                head=None,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))
        elif stage == "test":
            self.test_dataset = data.dataset_from_sharded_hdf5(
                self.training_config["datadir"] + "/test/",
                r_max=self.r_max,
                z_table=self.z_table,
                heads=None,
                head=None,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return get_data_loader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return get_data_loader(
            self.val_dataset,
            batch_size=self.training_config["bz"] * 3,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return

    @torch.enable_grad()
    def compute_loss(self, batch):
        res = self.potential.forward(batch.to(self.device), training=self.training)
        hat_ae = res["energy"].to(self.device)
        hat_forces = res["forces"].to(self.device)
        ae = batch.energy.to(self.device)
        forces = batch.forces.to(self.device)

        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        # fcosloss = 1 - self.cosineEval(hat_forces, forces)
        info = {
            "MAE_E": self.MAEEval(hat_ae, ae).item(),
            "MAE_F": self.MAEEval(hat_forces, forces).item(),
            "MAPE_E": self.MAPEEval(hat_ae, ae).item(),
            "MAPE_F": self.MAPEEval(hat_forces, forces).item(),
            "MAE_Fcos": 1
            - self.cosineEval(hat_forces.detach().cpu(), forces.detach().cpu()),
            "Loss_E": eloss.item(),
            "Loss_F": floss.item(),
        }

        loss = eloss + 100 * floss
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)
        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        return loss

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def validation_epoch_end(self, val_step_outputs):
        val_epoch_metrics = average_over_batch_metrics(val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 2 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = utils_diff.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
