from pysisyphus.constants import AU2EV, BOHR2ANG
from mace.calculators import mace_off_finetuned, mace_off
import torch

def get_mace_calculator(device="cpu", ver='finetuned'):
    # Disable CUDA if using CPU to prevent CUDA initialization errors
    if device == "cpu":
        # Temporarily disable CUDA to prevent initialization errors
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    
    try:
        if ver == 'finetuned':
            return mace_off_finetuned(device=device, model='/root/ReactBench/ckpt/mace.ckpt')
        elif ver == 'pretrain':
            return mace_off(model="medium", default_dtypes='float64') 
    finally:
        # Restore original CUDA availability check
        if device == "cpu":
            torch.cuda.is_available = original_cuda_available


class MACEMLFF:
    """MACE calculator for pysisyphus"""
    
    def __init__(self, device="cpu", ver='finetuned'):
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
        self.model = get_mace_calculator(device=device, ver=ver)
    
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
        hessian = self.model.get_hessian(atoms=molecule).reshape(molecule.get_number_of_atoms()*3,\
                                                                    molecule.get_number_of_atoms()*3) / AU2EV * BOHR2ANG * BOHR2ANG
        energy = molecule.get_potential_energy() / AU2EV 
        
        results = {
            "energy": energy,
            "hessian": hessian,
        }
        return results




