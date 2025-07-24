from pysisyphus.constants import AU2EV, BOHR2ANG
from mace.calculators import mace_off_finetuned, mace_off
import torch
import os


def get_mace_calculator(device="cpu", ver="finetuned", ckpt_path=None):
    # Disable CUDA if using CPU to prevent CUDA initialization errors
    if device == "cpu":
        # Temporarily disable CUDA to prevent initialization errors
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

    try:
        if ckpt_path is None:
            this_file_dir = os.path.dirname(os.path.abspath(__file__))
            proj_root_dir = os.path.dirname(this_file_dir)
            ckpt_path = f"{proj_root_dir}/ckpt/mace.ckpt"
        if ver == "finetuned":
            return mace_off_finetuned(device=device, model=ckpt_path)
        elif ver == "pretrain":
            return mace_off(model="medium", default_dtypes="float64")
    finally:
        # Restore original CUDA availability check
        if device == "cpu":
            torch.cuda.is_available = original_cuda_available


class MACEMLFF:
    """MACE calculator for pysisyphus"""

    def __init__(self, device="cpu", ver="finetuned", ckpt_path=None):
        """
        Initialize MACE calculator

        Parameters
        ----------
        device : str
            Device to run calculations on ('cpu' or 'cuda')
        ver : str
            Version of MACE model to use ('finetuned' or 'pretrain')
        """
        self.device = device
        self.model = get_mace_calculator(device=device, ver=ver, ckpt_path=ckpt_path)

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

        molecule.calc = self.model
        hessian = (
            self.model.get_hessian(atoms=molecule).reshape(
                molecule.get_number_of_atoms() * 3, molecule.get_number_of_atoms() * 3
            )
            / AU2EV
            * BOHR2ANG
            * BOHR2ANG
        )
        energy = molecule.get_potential_energy() / AU2EV

        results = {
            "energy": energy,
            "hessian": hessian,
        }
        return results
