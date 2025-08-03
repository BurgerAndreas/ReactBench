from pysisyphus.constants import AU2EV, BOHR2ANG
import os
import torch
import yaml
from typing import Optional
import sys

from ReactBench.Calculators._utils import compute_hessian

# Import required modules for Equiformer
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers

# Try to import Equiformer dependencies
from ocpmodels.common.relaxation.ase_utils import ase_atoms_to_torch_geometric
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props


class EquiformerCalculator(Calculator):
    """
    Equiformer ASE Calculator for ReactBench
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Equiformer calculator.

        Args:
            ckpt_path: Path to the trained Equiformer checkpoint file
            device: Optional device specification (defaults to auto-detect)
            config_path: Path to the model config file
            **kwargs: Additional keyword arguments for parent Calculator class
        """

        Calculator.__init__(self, **kwargs)

        # this is where all the calculated properties are stored
        self.results = {}
        
        _args = {
            "ckpt_path": ckpt_path,
            "device": device,
            "config_path": config_path,
            **kwargs,
        }
        print(f"{__file__} {self.__class__.__name__} got args: \n{_args}")

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        root_dir = os.environ.get("REACTBENCH_PATH", None)
        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root_dir = os.path.dirname(root_dir)

        # Load model
        # fix ckpt path
        if ckpt_path is None:
            ckpt_path = "ckpt/horm/eqv2.ckpt"
            ckpt_path = os.path.join(root_dir, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        # get the config to init the model
        if config_path == "auto":
            # get config from ckpt
            _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = _ckpt["hyper_parameters"]["model_config"]
        else:
            if config_path is None:
                # Try multiple possible locations for config file
                config_path = "../gad-ff/configs/equiformer_v2.yaml"
                config_path = os.path.join(root_dir, config_path)
            config_path = os.path.abspath(config_path)
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        model_config = config["model"]
        self.potential = EquiformerV2_OC20(**model_config)

        # Load model weights
        state_dict = torch.load(ckpt_path, weights_only=True)["state_dict"]
        state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
        self.potential.load_state_dict(state_dict, strict=False)

        self.potential.to(self.device)
        self.potential.eval()

        # Set implemented properties
        self.implemented_properties = ["energy", "forces", "hessian"]

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """
        Calculate properties for the given atoms.
        """
        Calculator.calculate(self, atoms)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(atoms)
        batch = batch.to(self.device)

        autograd = "hessian" in properties

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad=autograd)

        # Run prediction
        if autograd:
            with torch.enable_grad():
                energy, forces, _ = self.potential.forward(batch, eigen=False)
        else:
            with torch.no_grad():
                energy, forces, _ = self.potential.forward(batch, eigen=False)

        # Store results
        self.results = {}

        # Energy is per molecule, extract scalar value
        self.results["energy"] = float(energy.detach().cpu().item())

        if "hessian" in properties:
            hessian = compute_hessian(batch.pos, energy, forces).detach().cpu().numpy()
            self.results["hessian"] = hessian

        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach().cpu().numpy()


def get_equiformer_calculator(device="cpu", ckpt_path=None, config_path=None):
    """Get equiformer calculator for run_pygsm.py"""
    return EquiformerCalculator(
        device=device,
        ckpt_path=ckpt_path,
        config_path=config_path,
    )


class EquiformerMLFF:
    """Equiformer calculator for pysisyphus"""

    def __init__(
        self, device="cpu", ckpt_path=None, config_path=None, hessian_method="autograd"
    ):
        """
        Initialize Equiformer calculator

        Parameters
        ----------
        device : str
            Device to run calculations on ('cpu' or 'cuda')
        ckpt_path : str
            Path to checkpoint file
        config_path : str
            Path to config file
        """
        self.device = device
        self.hessian_method = hessian_method

        self.model = EquiformerCalculator(
            ckpt_path=ckpt_path,
            device=device,
            config_path=config_path,
        )

    def get_energy(self, molecule):
        """Get energy for pysisyphus interface"""
        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV

        results = {
            "energy": energy,
        }
        return results

    def get_forces(self, molecule):
        """Get forces for pysisyphus interface"""
        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV
        forces = molecule.get_forces() / AU2EV * BOHR2ANG

        results = {
            "energy": energy,
            "forces": forces.flatten(),
        }
        return results

    def get_hessian(self, molecule):
        """Get Hessian for pysisyphus interface"""
        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric(molecule)
        batch = batch.to(self.device)

        # Prepare batch with extra properties for autograd
        batch = compute_extra_props(batch, pos_require_grad=True)

        if self.hessian_method == "autograd":
            # Compute energy and forces with autograd
            with torch.enable_grad():
                energy, forces, _ = self.model.potential.forward(batch, eigen=False)

            # Use autograd to compute hessian
            hessian = (
                compute_hessian(batch.pos, energy, forces).detach().cpu().numpy()
                / AU2EV
                * BOHR2ANG
                * BOHR2ANG
            )
            energy = energy.item() / AU2EV
        elif self.hessian_method == "predict":
            raise NotImplementedError("Hessian prediction not implemented")
        else:
            raise ValueError(f"Invalid hessian method: {self.hessian_method}")

        results = {
            "energy": energy,
            "hessian": hessian,
        }
        return results
