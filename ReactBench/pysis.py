#!/bin/env python
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import subprocess
import os
import numpy as np
import re
import psutil
import logging
from ReactBench.utils.parsers import xyz_parse
from ReactBench.utils.taffi_functions import table_generator

logger = logging.getLogger(__name__)
THRESH = ["gau_loose", "gau", "gau_tight", "gau_vtight", "baker", "never"]
PYSCF_SOLVATION_OPTIONS = ["SMD", "DDCOSMO", "IEF-PCM", "C-PCM", "SS(V)PE", "COSMO"]
HA2KCALMOL = 627.509


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)  # Get all child processes
        for child in children:
            child.terminate()  # Gracefully terminate
        gone, still_alive = psutil.wait_procs(children, timeout=5)
        for child in still_alive:
            child.kill()  # Force kill remaining processes
        parent.kill()  # Finally, kill the parent process
    except psutil.NoSuchProcess:
        pass


class PYSIS:
    def __init__(
        self,
        input_geo,
        work_folder=os.getcwd(),
        jobname="pysis",
        jobtype="tsopt",
        coord_type="redund",
        nproc=1,
        mem=4000,
        exe=None,
        restart=False,
        dispersion="",
        functional="b3lyp",
        basis="6-31G*",
        charge=0,
        multiplicity=1,
        alpb=None,
        gbsa=None,
        solvation_model=None,
        solvent_epi=78.3553,
        calctype="mlff-leftnet",
        method=None,
        thresh="gau",
        hess=True,
        max_step=50,
        hess_step=3,
        hess_init=False,
        freeze_atoms=None,
        calc_kwargs={},
    ):
        """Initialize a pysisyphus job class.

        Args:
            input_geo (str): XYZ file containing input geometry. Full path recommended.
            work_folder (str): Working directory. Defaults to current directory.
            jobname (str): Name for the job. Defaults to 'pysis'.
            jobtype (str): Job type ('tsopt', 'irc', 'opt'). Defaults to 'tsopt'.
            coord_type (str): Coordinate system type. Defaults to 'redund'.
            nproc (int): Number of processors. Defaults to 1.
            mem (int): Memory per core in MB. Defaults to 4000.
            exe (str): Path to pysisyphus executable. Defaults to 'pysis'.
            restart (bool): Whether to restart from previous calculation. Defaults to False.
            dispersion (str): Whether to include dispersion correction. Defaults to ''.
            functional (str): DFT functional. Defaults to 'b3lyp'.
            basis (str): Basis set. Defaults to '6-31G*'.
            charge (int): Molecular charge. Defaults to 0.
            multiplicity (int): Spin multiplicity. Defaults to 1.
            alpb(str, optional): Whether to use ALPB solvation. Defaults to None.
            gbsa(str, optional): Whether to use GBSA solvation. Defaults to None.
            solvation_model(str, optional): Solvation model for pyscf
                - PCM: "IEF-PCM", "C-PCM", "SS(V)PE", "COSMO"
                - DDCOSMO
                - SMD
                Defaults to None.
            solvent_epi (float): Solvent dielectric constant. Defaults to 78.3553.
            calctype (str): Calculator type. Defaults to 'mlff-leftnet'.
            method (str, optional): Optimization method. Defaults to None.
            thresh (str): Convergence threshold. Defaults to 'gau'.
            hess (bool): Whether to calculate Hessian. Defaults to True.
            max_step (int): Maximum optimization steps. Defaults to 50.
            hess_step (int): Steps between Hessian recalc. Defaults to 3.
            hess_init (bool): Use initial Hessian. Defaults to False.
            freeze_atoms (list): List of atoms to freeze. Defaults to None.
            calc_kwargs (dict): Calculator-specific settings. Defaults to {}.
        """
        # turn the relative path to absolute path for input_geo
        if not os.path.isabs(input_geo):
            input_geo = os.path.abspath(input_geo)
        # turn the relative path to absolute path for work_folder
        if not os.path.isabs(work_folder):
            work_folder = os.path.abspath(work_folder)
        self.input_geo = input_geo
        self.work_folder = work_folder
        self.pysis_input = os.path.join(work_folder, f"pysis_{jobname}_{jobtype}_input.yaml")
        self.output = os.path.join(work_folder, f"pysis_{jobtype}_output.txt")
        self.errlog = os.path.join(work_folder, f"pysis_{jobname}-{jobtype}.err")
        self.nproc = int(nproc)
        self.mem = int(mem)
        self.jobname = jobname
        self.jobtype = jobtype
        self.restart = restart
        self.coord_type = coord_type
        self.charge = charge
        self.multiplicity = multiplicity
        self.functional = functional
        self.basis = basis
        self.alpb = alpb
        self.gbsa = gbsa
        self.solvation_model = solvation_model
        self.solvent_epi = solvent_epi
        self.dispersion = dispersion
        self.exe = exe if exe is not None else "pysis"
        self.calc_kwargs = calc_kwargs

        os.makedirs(self.work_folder, exist_ok=True)
        # create a pysis_input file
        self.generate_input(
            calctype=calctype,
            method=method,
            thresh=thresh,
            hess=hess,
            max_step=max_step,
            hess_step=hess_step,
            hess_init=hess_init,
            freeze_atoms=freeze_atoms,
            calc_kwargs=calc_kwargs,
        )
        # print(f"PYSIS job {self.jobname} created input file: {self.pysis_input}")

    def generate_calculator_settings(self, calctype="mlff-leftnet", calc_kwargs={}):
        """Generate calculator-specific settings for the input file.

        Args:
            calctype (str, optional): Calculator type ('mlff-*'). Defaults to 'mlff-leftnet'.

        Returns:
            bool: False if calculator type not supported, None otherwise.
        """
        with open(self.pysis_input, "a") as f:
            if calctype.startswith("mlff"):
                method = "-".join(calctype.split("-")[1:])
                f.write(
                    f"calc:\n type: mlff\n method: {method}\n pal: {self.nproc}\n mem: {self.mem}\n charge: {self.charge}\n mult: {self.multiplicity}\n"
                )
                # print(f"PYSIS job {self.jobname} {self.jobtype} got calc_kwargs: {calc_kwargs}")
                for key, value in calc_kwargs.items():
                    f.write(f" {key}: {value}\n")

            elif calctype == "pyscf":
                settings = {
                    "type": calctype,
                    "method": "dft",
                    "xc": self.functional,
                    "basis": self.basis,
                    "pal": self.nproc,
                    "mem": self.mem,
                    "charge": self.charge,
                    "mult": self.multiplicity,
                }

                if self.solvation_model:
                    if self.solvation_model.upper() not in PYSCF_SOLVATION_OPTIONS:
                        logger.warning(
                            f"Your solvation model is {self.solvation_model.upper()}, currently, only {PYSCF_SOLVATION_OPTIONS} are supported. Here used SMD instead"
                        )
                        self.solvation_model = "SMD"
                    settings["solvation_model"] = self.solvation_model
                    settings["solvent_epi"] = self.solvent_epi
                if self.dispersion:
                    settings["dispersion"] = self.dispersion

                f.write("calc:\n")
                for key, value in settings.items():
                    f.write(f" {key}: {value}\n")
                for key, value in calc_kwargs.items():
                    f.write(f" {key}: {value}\n")

            else:
                print("Supports for other packages are underway")
                return False

    def generate_job_settings(
        self,
        method=None,
        thresh="gau",
        max_step=50,
        hess=True,
        hess_step=3,
        hess_init=False,
        calc_kwargs={},
    ):
        """Generate job-specific settings for PYSIS calculation.

        Args:
            method (str, optional): Optimization method. Defaults to None.
                - TSOPT: 'rsirfo', 'rsprfo' (default), 'trim'
                - IRC: 'euler', 'eulerpc' (default), 'dampedvelocityverlet',
                       'gonzalezschlegel', 'lqa', 'imk', 'rk4'
                - OPT: 'rfo' (default)
            thresh (str, optional): Convergence threshold. Defaults to 'gau'.
                Options: 'gau_loose', 'gau', 'gau_tight', 'gau_vtight'
            max_step (int, optional): Maximum optimization steps. Defaults to 50.
            hess (bool, optional): Whether to calculate Hessian. Defaults to True.
            hess_step (int, optional): Steps between Hessian recalc. Defaults to 3.
            hess_init (bool, optional): Use initial Hessian. Defaults to False.
        """
        jobtype = self.jobtype.lower()
        if thresh.lower() not in THRESH:
            logger.warning(
                f"Your threshold is {thresh}, currently, only {THRESH} are supported. Here used gau instead"
            )
            thresh = "gau"

        with open(self.pysis_input, "a") as f:
            if jobtype == "tsopt":
                method = method or "rsprfo"
                settings = {
                    "type": method,
                    "do_hess": hess,
                    "thresh": thresh,
                    "max_cycles": max_step,
                    "trust_radius": 0.2,
                }
                if hess:
                    settings["hessian_recalc"] = hess_step

                f.write("tsopt:\n")
                for key, value in settings.items():
                    f.write(f" {key}: {value}\n")

            elif jobtype == "irc":
                method = method or "eulerpc"
                f.write(
                    f"irc:\n type: {method}\n forward: True\n backward: True\n downhill: False\n"
                )
                if hess_init:
                    f.write(f" hessian_init: {hess_init}\n")
                f.write(
                    f"endopt:\n fragments: False\n do_hess: False\n thresh: {thresh}\n max_cycles: 50"
                )

            elif jobtype == "opt":
                method = method or "rfo"
                settings = {
                    "type": method,
                    "do_hess": hess,
                    "thresh": thresh,
                    "max_cycles": max_step,
                }

                f.write("opt:\n")
                for key, value in settings.items():
                    f.write(f" {key}: {value}\n")

            else:
                print("Supports for other job types are underway")
                return False

    def generate_input(
        self,
        calctype="mlff-leftnet",
        method=None,
        thresh="gau",
        hess=True,
        max_step=50,
        hess_step=3,
        hess_init=False,
        freeze_atoms=None,
        calc_kwargs={},
    ):
        """Create a PYSIS input file based on settings.

        Args:
            calctype (str, optional): Calculator type. Defaults to 'mlff-leftnet'.
            method (str, optional): Optimization method. Defaults to None.
            thresh (str, optional): Convergence threshold. Defaults to 'gau'.
            hess (bool, optional): Whether to calculate Hessian. Defaults to True.
            max_step (int, optional): Maximum optimization steps. Defaults to 50.
            hess_step (int, optional): Steps between Hessian recalc. Defaults to 3.
            hess_init (bool, optional): Use initial Hessian. Defaults to False.
            freeze_atoms (list, optional): List of atoms to freeze. Defaults to None.
            calc_kwargs (dict, optional): Calculator-specific settings. Defaults to {}.
        """
        freeze_atoms = freeze_atoms or []

        # Write geometry section
        with open(self.pysis_input, "w") as f:
            geom_settings = {"fn": self.input_geo}

            if self.coord_type.lower() in ["cart", "redund", "dlc", "tric"]:
                geom_settings["type"] = self.coord_type
            else:
                logger.warning(
                    f"Your coordinate type is {self.coord_type}, currently, only cart, redund, dlc and tric are supported. Here used redund instead"
                )
                geom_settings["type"] = "redund"

            if freeze_atoms:
                geom_settings["freeze_atoms"] = freeze_atoms

            f.write("geom:\n")
            for key, value in geom_settings.items():
                f.write(f" {key}: {value}\n")

        # Generate calculator and job settings
        self.generate_calculator_settings(calctype=calctype, calc_kwargs=calc_kwargs)
        self.generate_job_settings(
            method=method,
            thresh=thresh,
            hess=hess,
            max_step=max_step,
            hess_step=hess_step,
            hess_init=hess_init,
        )

    def execute(self, timeout=3600, cleanup=True):
        """Execute a PYSIS calculation.

        Args:
            timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
            cleanup: cleanup qm_calcs and h5 files
        Returns:
            str: Status message indicating job completion or failure.
        """

        # Check job status before executing
        if self.calculation_terminated_normally():
            msg = f"PYSIS job {self.jobname} has been finished, skip this job..."
            print(msg)
            return msg

        if os.path.isfile(self.output) and not self.restart:
            msg = f"PYSIS job {self.jobname} fails, skip this job..."
            print(msg)
            return msg

        try:
            os.chdir(self.work_folder)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(self.nproc)

            print(f"running PYSIS job {self.jobname}")

            with open(self.output, "w") as stdout, open(self.errlog, "w") as stderr:
                process = subprocess.Popen(
                    f"{self.exe} {self.pysis_input}",
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=env,
                    shell=True,
                    cwd=self.work_folder,
                )

                # Wait for the process to complete or timeout
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    result = subprocess.CompletedProcess(
                        args=f"{self.exe} {self.pysis_input}",
                        returncode=process.returncode,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except subprocess.TimeoutExpired:
                    kill_process_tree(process.pid)
                    result = subprocess.CompletedProcess(
                        args=f"{self.exe} {self.pysis_input}",
                        returncode=1,
                        stdout="",
                        stderr=f"PYSIS job {self.jobname} timed out after {timeout} seconds",
                    )

        finally:
            if "process" in locals() and process.poll() is None:
                kill_process_tree(process.pid)

            if cleanup:
                # Cleanup temporary files
                tmp_scratch = f"{self.work_folder}/qm_calcs"
                if os.path.exists(tmp_scratch):
                    for file in os.listdir(tmp_scratch):
                        os.remove(os.path.join(tmp_scratch, file))

                # find all h5 (besides final_hessian)
                remove_h5s = [
                    h5 for h5 in os.listdir(self.work_folder) if h5.endswith(".h5")
                ]
                for h5 in remove_h5s:
                    if "final_hessian" not in h5:
                        os.remove(os.path.join(self.work_folder, h5))

                # remove useless log files
                log_files = ["optimizer.log", "pysisyphus.log", "internal_coords.log"]
                for log in log_files:
                    log_file = os.path.join(self.work_folder, log)
                    if os.path.isfile(log_file):
                        os.remove(log_file)

        if result.returncode == 0:
            msg = f"PYSIS job {self.jobname} is finished."
        else:
            msg = f"Command failed for PYSIS job {self.jobname}, check job log file for detailed information"
            with open(self.output, "a") as f:
                f.write("\nError termination of PYSIS...\n")

        print(msg)
        return msg

    def calculation_terminated_with_error(self) -> bool:
        """Check if the calculation terminated with error.

        Returns:
            bool: True if error found in output, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Error termination of" in line or "Aborting!" in line:
                    return True

        return False

    def calculation_terminated_normally(self) -> bool:
        """Check if the calculation terminated normally.

        Returns:
            bool: True if calculation completed successfully, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "pysisyphus run took" in line:
                    return True

        return False

    def optimization_converged(self) -> bool:
        """Check if the optimization converged.

        Returns:
            bool: True if optimization converged, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        if not self.calculation_terminated_normally():
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Converged!" in line:
                    return True

        return False

    def optimization_success(self) -> bool:
        """Check if optimization converged and structure maintained connectivity.

        Returns:
            bool: True if optimization successful and structure valid, False otherwise
        """
        if not self.optimization_converged():
            return False

        elements, initial_geom = xyz_parse(self.input_geo)
        _, final_geom = self.get_opted_geo()

        adj_mat_initial = table_generator(elements, initial_geom)
        adj_mat_final = table_generator(elements, final_geom)

        if np.array_equal(adj_mat_initial, adj_mat_final):
            return True

        # Check if differences only involve metal atoms
        metal_atoms = {"Zn", "Mg", "Li", "Si", "Ag"}
        rows, cols = np.where(adj_mat_initial != adj_mat_final)

        for i, j in zip(rows, cols):
            if elements[i] not in metal_atoms and elements[j] not in metal_atoms:
                return False

        return True

    def is_true_ts(self) -> bool:
        """Check if structure has exactly one significant imaginary frequency.

        Returns:
            bool: True if exactly one imaginary frequency < -10 cm^-1, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        if not self.calculation_terminated_normally():
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Imaginary frequencies:" in line:
                    freqs = [float(x) for x in re.findall(r"-?\d+\.?\d*", line)]
                    return len(freqs) == 1 and freqs[0] < -10

        return False

    def get_energy(self) -> float:
        """Get single point energy from the output file.

        Returns:
            float: Energy value if found, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "energy:" in line:
                    return float(line.split()[-2])

        return False

    def get_final_structure(self):
        """Get the final optimized geometry.

        Returns:
            tuple: (elements, coordinates) if successful, (False, []) otherwise
        """
        elements, geometry = self.get_final_ts()
        if elements:
            return elements, geometry

        elements, geometry = self.get_opted_geo()
        if elements:
            return elements, geometry

        return False, []

    def get_opted_geo(self):
        """Get the final optimized geometry.

        Returns:
            tuple: (elements, coordinates) if successful, (False, []) otherwise
        """
        xyz_file = f"{self.work_folder}/final_geometry.xyz"
        if os.path.exists(xyz_file):
            return xyz_parse(xyz_file)
        return False, []

    def get_final_ts(self):
        """Get the final transition state geometry.

        Returns:
            tuple: (elements, coordinates) if successful, (False, []) otherwise
        """
        ts_files = [
            f"{self.work_folder}/ts_opt.xyz",
            f"{self.work_folder}/ts_final_geometry.xyz",
        ]

        for ts_file in ts_files:
            if os.path.exists(ts_file):
                return xyz_parse(ts_file)

        return False, []

    def analyze_IRC(self, return_traj: bool = False):
        """Analyze IRC calculation results.

        Args:
            return_traj: Whether to return the full IRC trajectory

        Returns:
            If return_traj is False:
                tuple: (elements, reactant_geom, product_geom, ts_geom, barrier_left, barrier_right)
            If return_traj is True:
                tuple: (elements, reactant_geom, product_geom, ts_geom, barrier_left, barrier_right, trajectory)

            Barriers are in kcal/mol
        """
        # Get barriers from output
        mols, info = xyz_parse(
            f"{self.work_folder}/finished_irc.trj", multiple=True, return_info=True
        )
        energies = [float(i) for i in info]

        e_ts = max(energies)
        e_r = energies[0]
        e_p = energies[-1]
        barrier_left = (e_ts - e_r) * HA2KCALMOL
        barrier_right = (e_ts - e_p) * HA2KCALMOL

        # Load geometries
        elements, ts_geom = xyz_parse(self.input_geo)
        reactant_geom = mols[0][1]
        product_geom = mols[-1][1]

        if not return_traj:
            return (
                elements,
                reactant_geom,
                product_geom,
                ts_geom,
                barrier_left,
                barrier_right,
                e_ts,
            )

        return (
            elements,
            reactant_geom,
            product_geom,
            ts_geom,
            barrier_left,
            barrier_right,
            e_ts,
            mols,
        )
