from pysisyphus.constants import AU2EV, BOHR2ANG
import os
import torch
import yaml
from typing import Optional
import sys
from typing import Callable, Literal, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ReactBench.Calculators._utils import compute_hessian
from pysisyphus.calculators.Calculator import Calculator as PysisCalculator

# Import required modules for Equiformer
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.data import atomic_numbers
import ase
import ase.atoms

# Try to import Equiformer dependencies
try:
    from ocpmodels.common.relaxation.ase_utils import (
        ase_atoms_to_torch_geometric_hessian,
        coord_atoms_to_torch_geometric_hessian,
    )
    from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
    from nets.prediction_utils import compute_extra_props

    equiformer_available = True
except ImportError:
    equiformer_available = False


class EquiformerCalculator(ASECalculator):
    """
    Equiformer ASE Calculator for ReactBench.
    Everything in Angstrom and eV.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        hessian_method: Optional[str] = "autograd",
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
        if not equiformer_available:
            raise ImportError(
                f"Equiformer is not available. Loading imports failed in {__file__}."
            )

        ASECalculator.__init__(self, **kwargs)

        _args = {
            "ckpt_path": ckpt_path,
            "device": device,
            "config_path": config_path,
            "hessian_method": hessian_method,
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
        if ckpt_path in [None, "None", "none", "null"]:
            ckpt_path = "ckpt/horm/eqv2.ckpt"
            ckpt_path = os.path.join(root_dir, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        # get the config to init the model
        if config_path == "auto":
            # get config from ckpt
            _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_config = _ckpt["hyper_parameters"]["model_config"]
        else:
            if config_path in [None, "None", "none", "null"]:
                # Try multiple possible locations for config file
                config_path = "../gad-ff/configs/equiformer_v2.yaml"
                config_path = os.path.join(root_dir, config_path)
            config_path = os.path.abspath(config_path)
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            model_config = config["model"]
        self.potential = EquiformerV2_OC20(**model_config)

        # Load model weights
        state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
        state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
        self.potential.load_state_dict(state_dict, strict=False)

        self.potential.to(self.device)
        self.potential.eval()

        assert hessian_method in ["autograd", "predict"], (
            f"Invalid hessian method: {hessian_method}"
        )
        self.hessian_method = hessian_method

        # Set implemented properties
        self.implemented_properties = ["energy", "forces", "hessian"]

        self.reset()

    def reset(self):
        # this is where all the calculated properties are stored
        self.results = {}
        self.cnt_hessian_autograd = 0
        self.cnt_hessian_predict = 0
        super().reset()

    def calculate(
        self, atoms=None, properties=None, system_changes=None, hessian_method=None
    ):
        """
        Calculate properties for the given atoms.
        """
        if hessian_method is None:
            hessian_method = self.hessian_method

        ASECalculator.calculate(self, atoms)

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric_hessian(
            atoms,
            cutoff=self.potential.cutoff,
            max_neighbors=self.potential.max_neighbors,
            use_pbc=self.potential.use_pbc,
        )
        batch = batch.to(self.device)

        do_hessian = "hessian" in properties
        do_autograd = do_hessian and (hessian_method == "autograd")

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad=do_autograd)

        # Store results
        self.results = {}

        # Run prediction
        N = batch.pos.shape[0]
        if do_autograd:
            with torch.enable_grad():
                energy, forces, _ = self.potential.forward(
                    batch, eigen=False, otf_graph=True
                )
                hessian = (
                    compute_hessian(batch.pos, energy, forces).detach().cpu().numpy()
                )
                self.results["hessian"] = hessian.reshape(N * 3, N * 3)
                self.cnt_hessian_autograd += 1
        else:
            with torch.no_grad():
                energy, forces, out = self.potential.forward(
                    batch, eigen=False, hessian=do_hessian
                )
                if do_hessian:
                    self.results["hessian"] = (
                        out["hessian"].detach().cpu().numpy().reshape(N * 3, N * 3)
                    )
                    self.cnt_hessian_predict += 1

        # Energy is per molecule, extract scalar value
        self.results["energy"] = float(energy.detach().cpu().item())
        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach().cpu().numpy().reshape(-1)


def get_equiformer_calculator(device="cpu", ckpt_path=None, config_path=None, **kwargs):
    """Get equiformer calculator for run_pygsm.py"""
    return EquiformerCalculator(
        device=device,
        ckpt_path=ckpt_path,
        config_path=config_path,
        **kwargs,
    )


class EquiformerMLFF:
    """Equiformer calculator for pysisyphus.
    Everything in Bohr and Hartree. Unit conversion taken from LeftNetMLFF.
    """

    def __init__(
        self,
        device="cpu",
        ckpt_path=None,
        config_path=None,
        hessian_method="autograd",
        **kwargs,
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
            hessian_method=hessian_method,
            **kwargs,
        )
        self.reset()

    def reset(self):
        self.cnt_hessian_autograd = 0
        self.cnt_hessian_predict = 0
        self.model.reset()

    # TODO: why are get_energy and get_forces different than get_hessian?
    def get_energy(self, molecule: ase.atoms.Atoms):
        """Get energy for pysisyphus interface"""
        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV
        assert energy is not None, f"Energy is None for {molecule}"

        results = {
            "energy": energy,
        }
        return results

    def get_forces(self, molecule: ase.atoms.Atoms):
        """Get forces for pysisyphus interface"""
        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV
        forces = molecule.get_forces() / AU2EV * BOHR2ANG

        results = {
            "energy": energy,
            "forces": forces.flatten(),
        }
        return results

    def get_hessian(self, molecule: ase.atoms.Atoms, hessian_method=None):
        """Get Hessian for pysisyphus interface"""

        if hessian_method is None:
            hessian_method = self.hessian_method

        with_grad = hessian_method == "autograd"

        # Convert ASE atoms to torch_geometric format
        batch = ase_atoms_to_torch_geometric_hessian(
            molecule,
            cutoff=self.model.potential.cutoff,
            max_neighbors=self.model.potential.max_neighbors,
            use_pbc=self.model.potential.use_pbc,
            with_grad=with_grad,
        )
        batch = batch.to(self.device)

        # Prepare batch with extra properties for autograd
        batch = compute_extra_props(batch)

        if hessian_method == "autograd":
            # Compute energy and forces with autograd
            with torch.enable_grad():
                # batch.pos.requires_grad = True # already set in ase_atoms_to_torch_geometric_hessian
                energy, forces, _ = self.model.potential.forward(
                    batch,
                    eigen=False,
                    otf_graph=True,
                )
                # Use autograd to compute hessian
                hessian = compute_hessian(
                    coords=batch.pos,
                    energy=energy,
                    forces=forces,  # allow_unused=True
                )
            self.model.cnt_hessian_autograd += 1
            self.cnt_hessian_autograd += 1

        elif hessian_method == "predict":
            with torch.no_grad():
                energy, forces, out = self.model.potential.forward(
                    batch, eigen=False, hessian=True
                )
                hessian = out["hessian"]
            self.model.cnt_hessian_predict += 1
            self.cnt_hessian_predict += 1
        else:
            raise ValueError(f"Invalid hessian method: {hessian_method}")

        N = batch.pos.shape[0]
        # ev, angstrom -> hartree (au), bohr
        energy = energy.item() / AU2EV
        forces = forces.detach().cpu().numpy().reshape(-1) / AU2EV * BOHR2ANG
        hessian = hessian.detach().cpu().numpy()
        hessian = hessian.reshape(N * 3, N * 3)
        hessian = hessian / AU2EV * BOHR2ANG * BOHR2ANG

        results = {
            "energy": energy,
            "forces": forces,
            "hessian": hessian,
        }
        return results


#########################################################################################


# pysisyphus.linalg
def finite_difference_hessian(
    coords: NDArray[float],
    grad_func: Callable[[NDArray[float]], NDArray[float]],
    step_size: float = 1e-2,
    acc: Literal[2, 4] = 2,
    callback: Optional[Callable] = None,
) -> NDArray[float]:
    """Numerical Hessian from central finite gradient differences.

    See central differences in
      https://en.wikipedia.org/wiki/Finite_difference_coefficient
    for the different accuracies.
    """
    if callback is None:

        def callback(*args):
            pass

    accuracies = {
        2: ((-0.5, -1), (0.5, 1)),  # 2 calculations
        4: ((1 / 12, -2), (-2 / 3, -1), (2 / 3, 1), (-1 / 12, 2)),  # 4 calculations
    }
    accs_avail = list(accuracies.keys())
    assert acc in accs_avail

    size = coords.size
    fd_hessian = np.zeros((size, size))
    zero_step = np.zeros(size)

    coeffs = accuracies[acc]
    for i, _ in enumerate(coords):
        step = zero_step.copy()
        step[i] = step_size

        def get_grad(factor, displ, j):
            displ_coords = coords + step * displ
            callback(i, j)
            grad = grad_func(displ_coords)
            return factor * grad

        grads = [get_grad(factor, displ, j) for j, (factor, displ) in enumerate(coeffs)]
        fd = np.sum(grads, axis=0) / step_size
        fd_hessian[i] = fd

    # Symmetrize
    fd_hessian = (fd_hessian + fd_hessian.T) / 2
    return fd_hessian


SYMBOL_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


class PysisEquiformer(PysisCalculator):
    conf_key = "mlff"

    def __init__(
        self,
        device="cpu",
        ckpt_path=None,
        config_path=None,
        hessian_method="autograd",
        **kwargs,
    ):
        """MLFF calculator.

        Wrapper for running energy, gradient and Hessian calculations by
        different MLFF.

        Parameters
        ----------
        method: str
            select a MLFF from calculators in ReactBench

        mem : int
            Mememory per core in MB.
        quiet : bool, optional
            Suppress creation of log files.
        """
        super().__init__(**kwargs)

        if not equiformer_available:
            raise ImportError(
                f"Equiformer is not available. Loading imports failed in {__file__}."
            )

        _args = {
            "ckpt_path": ckpt_path,
            "device": device,
            "config_path": config_path,
            "hessian_method": hessian_method,
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
        if ckpt_path in [None, "None", "none", "null"]:
            ckpt_path = "ckpt/horm/eqv2.ckpt"
            ckpt_path = os.path.join(root_dir, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        # get the config to init the model
        if config_path == "auto":
            # get config from ckpt
            _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_config = _ckpt["hyper_parameters"]["model_config"]
        else:
            if config_path in [None, "None", "none", "null"]:
                # Try multiple possible locations for config file
                config_path = "../gad-ff/configs/equiformer_v2.yaml"
                config_path = os.path.join(root_dir, config_path)
            config_path = os.path.abspath(config_path)
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            model_config = config["model"]
        self.potential = EquiformerV2_OC20(**model_config)

        # Load model weights
        state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
        state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
        self.potential.load_state_dict(state_dict, strict=False)

        self.potential.to(self.device)
        self.potential.eval()

        assert hessian_method in ["autograd", "predict"], (
            f"Invalid hessian method: {hessian_method}"
        )
        self.hessian_method = hessian_method

        # Set implemented properties
        self.implemented_properties = ["energy", "forces", "hessian"]

        self.reset()

    def reset(self):
        # this is where all the calculated properties are stored
        self.results = {}
        self.cnt_energy = 0
        self.cnt_forces = 0
        self.cnt_hessian_autograd = 0
        self.cnt_hessian_predict = 0
        self.cnt_hessian_num = 0
        super().reset()

    def prepare_batch(self, atoms, coords, with_grad=False):
        # Convert ASE atoms to torch_geometric format
        # str symbols -> int z
        N = len(atoms)
        z = [SYMBOL_TO_Z[a] for a in atoms]
        batch = coord_atoms_to_torch_geometric_hessian(
            coords.reshape(N, 3) * BOHR2ANG,
            z,
            cutoff=self.potential.cutoff,
            max_neighbors=self.potential.max_neighbors,
            use_pbc=self.potential.use_pbc,
            with_grad=with_grad,
        )
        batch = batch.to(self.device)

        # Prepare batch with extra properties for autograd
        batch = compute_extra_props(batch)
        return batch

    def store_and_track(self, results, func, atoms, coords):
        prepare_kwargs = {}
        if self.track:
            self.store_overlap_data(atoms, coords)
            if self.track_root():
                # Redo the calculation with the updated root
                results = func(atoms, coords, **prepare_kwargs)
        return results

    def get_energy(self, atoms, coords):
        batch = self.prepare_batch(atoms, coords)
        energy, forces, _ = self.potential.forward(batch)
        self.cnt_energy += 1
        return {
            "energy": energy.item() / AU2EV,
            "forces": forces.detach().cpu().numpy().reshape(-1) / AU2EV * BOHR2ANG,
        }

    def get_forces(self, atoms, coords):
        batch = self.prepare_batch(atoms, coords)
        energy, forces, _ = self.potential.forward(batch)
        self.cnt_forces += 1
        return {
            "energy": energy.item() / AU2EV,
            "forces": forces.detach().cpu().numpy().reshape(-1) / AU2EV * BOHR2ANG,
        }

    def get_hessian(
        self, atoms: list[str], coords: NDArray[float], hessian_method=None, **kwargs
    ):
        if hessian_method is None:
            hessian_method = self.hessian_method
        with_grad = hessian_method == "autograd"

        batch = self.prepare_batch(atoms, coords, with_grad)

        if hessian_method == "autograd":
            # Compute energy and forces with autograd
            with torch.enable_grad():
                # batch.pos.requires_grad = True # already set in ase_atoms_to_torch_geometric_hessian
                energy, forces, _ = self.potential.forward(
                    batch,
                    eigen=False,
                    otf_graph=True,
                )
                # Use autograd to compute hessian
                hessian = compute_hessian(
                    coords=batch.pos,
                    energy=energy,
                    forces=forces,  # allow_unused=True
                )
            self.cnt_hessian_autograd += 1

        elif hessian_method == "predict":
            with torch.no_grad():
                energy, forces, out = self.potential.forward(
                    batch, eigen=False, hessian=True
                )
                hessian = out["hessian"]
            self.cnt_hessian_predict += 1
        else:
            raise ValueError(f"Invalid hessian method: {hessian_method}")

        N = batch.pos.shape[0]
        # ev, angstrom -> hartree (au), bohr
        energy = energy.item() / AU2EV
        forces = forces.detach().cpu().numpy().reshape(-1) / AU2EV * BOHR2ANG
        hessian = hessian.detach().cpu().numpy()
        hessian = hessian.reshape(N * 3, N * 3)
        hessian = hessian / AU2EV * BOHR2ANG * BOHR2ANG

        results = {
            "energy": energy,
            "forces": forces,
            "hessian": hessian,
        }
        return results

    def get_num_hessian(self, atoms, coords, **prepare_kwargs):
        """
        geom.calculator.get_num_hessian(geom.atoms, geom._coords)
        """
        print(f"{__file__} {self.__class__.__name__} Calculating numerical Hessian.")
        self.cnt_hessian_num += 1
        results = self.get_energy(atoms, coords, **prepare_kwargs)

        def grad_func(coords):
            results = self.get_forces(atoms, coords, **prepare_kwargs)
            gradient = -results["forces"]
            return gradient

        def callback(i, j):
            self.log(f"Displacement {j} of coordinate  {i}")

        _num_hess_kwargs = {
            "step_size": 0.005,
            # Central difference by default
            "acc": 2,
        }
        _num_hess_kwargs.update(self.num_hess_kwargs)

        fd_hessian = finite_difference_hessian(
            coords,
            grad_func,
            callback=callback,
            **_num_hess_kwargs,
        )
        results["hessian"] = fd_hessian
        return results

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"PysisEquiformer({__file__})"
