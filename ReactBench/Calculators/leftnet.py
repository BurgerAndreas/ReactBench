from pysisyphus.constants import AU2EV, BOHR2ANG
import os

from ReactBench.Calculators._utils import compute_hessian
from ReactBench.MLIP.leftnet.oa_reactdiff.trainer.calculator import (
    LeftNetCalculator,
    mols_to_batch,
)


# get leftnet calculator for run_pygsm.py
def get_leftnet_calculator(device="cpu", use_autograd=True, ckpt_path=None, **kwargs):
    if ckpt_path is None:
        # ReactBench/Calculators/
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # ReactBench/
        proj_root_dir = os.path.dirname(os.path.dirname(this_file_dir))
        if use_autograd:
            ckpt_path = f"{proj_root_dir}/ckpt/leftnet.ckpt"
        else:
            ckpt_path = f"{proj_root_dir}/ckpt/leftnet-df.ckpt"
        print(f"Using default checkpoint for LeftNet: {ckpt_path}")
    print(f"{__file__} get_leftnet_calculator ignoring kwargs: \n{kwargs}")
    return LeftNetCalculator(weight=ckpt_path, device=device, use_autograd=use_autograd)


# get leftnet mlff for pysisyphus
class LeftNetMLFF:
    """LeftNet calculator for pysisyphus"""

    def __init__(self, device="cpu", use_autograd=True, ckpt_path=None, **kwargs):
        """
        Initialize LeftNet calculator

        Parameters
        ----------
        device : str
            Device to run calculations on ('cpu' or 'cuda')
        use_autograd : bool
            Whether to use autograd for force calculation
        """
        print(f"{__file__} LeftNetMLFF ignoring kwargs: \n{kwargs}")
        self.device = device
        self.use_autograd = use_autograd
        print(f"{__file__} LeftNetMLFF use_autograd={use_autograd}")
        if ckpt_path is None:
            this_file_dir = os.path.dirname(os.path.abspath(__file__))
            proj_root_dir = os.path.dirname(os.path.dirname(this_file_dir))
            # ckpt are from ReactBench paper, not HORM
            if use_autograd:
                ckpt_path = f"{proj_root_dir}/ckpt/leftnet.ckpt"
            else:
                ckpt_path = f"{proj_root_dir}/ckpt/leftnet-df.ckpt"
            print(f"{__file__} LeftNetMLFF using default checkpoint: {ckpt_path}")
        self.model = LeftNetCalculator(
            weight=ckpt_path,
            device=device,
            use_autograd=use_autograd,
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

        data = mols_to_batch([molecule]).to(self.device)

        # compute energy and force
        if self.use_autograd:
            energy, forces = self.model.model.forward_autograd(data)
        else:
            energy, forces = self.model.model.forward(data)

        # use autograd of force to compute hessian
        hessian = (
            compute_hessian(data.pos, energy, forces).detach().cpu().numpy()
            / AU2EV
            * BOHR2ANG
            * BOHR2ANG
        )
        energy = energy.item() / AU2EV

        results = {
            "energy": energy,
            "hessian": hessian,
        }
        return results
