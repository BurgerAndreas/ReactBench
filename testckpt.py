import torch
import os


ckpt_path = "ckpt/horm/eqv2.ckpt"

_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(_ckpt["hyper_parameters"]["model_config"].keys())