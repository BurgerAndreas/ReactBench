#!/bin/env python
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)
import subprocess
import os
import time

from ReactBench.utils.parsers import xyz_parse


class PYGSM:
    def __init__(
        self,
        input_geo,
        work_folder=os.getcwd(),
        calc="leftnet",
        jobname="gsmjob",
        jobid=1,
        nprocs=1,
        num_nodes=9,
        max_gsm_iters=100,
        max_opt_steps=3,
        add_node_tol=0.1,
        conv_tol=0.0005,
        reactant_geom_fixed=False,
        product_geom_fixed=False,
        dmax=0.1,
        restart=False,
        source_path=None,
        python_exe="python",
        multiplicity=1,
        charge=0,
        # Added Andreas
        ckpt_path=None,
        config_path=None,
        hessian_method="autograd",
        device="cpu",
    ):
        """Initialize a pyGSM job class.

        Args:
            input_geo (str): XYZ file containing input geometry of reactant and product
            work_folder (str, optional): Working directory. Defaults to current directory.
            calc (str, optional): Calc engine. Defaults to 'leftnet'.
            jobname (str, optional): Name for the job. Defaults to 'gsmjob'.
            jobid (int, optional): Numeric ID for the job. Defaults to 1.
            nprocs (int, optional): Number of processors to use. Defaults to 1.
            num_nodes (int, optional): Number of nodes in string. Defaults to 9.
            max_gsm_iters (int, optional): Maximum GSM iterations. Defaults to 100.
            max_opt_steps (int, optional): Maximum optimization steps. Defaults to 3.
            add_node_tol (float, optional): Node addition tolerance. Defaults to 0.1.
            conv_tol (float, optional): Convergence tolerance. Defaults to 0.0005.
            reactant_geom_fixed (bool, optional): Fix reactant geometry. Defaults to False.
            product_geom_fixed (bool, optional): Fix product geometry. Defaults to False.
            dmax (float, optional): Maximum step size. Defaults to 0.1.
            restart (bool, optional): Whether to restart from previous calculation. Defaults to False.
            source_path (str, optional): Path to pyGSM source files. Defaults to package directory.
            python_exe (str, optional): Python executable to use. Defaults to 'python'.
        """
        self.input_geo = input_geo
        self.jobid = jobid
        self.nprocs = nprocs
        self.work_folder = work_folder
        self.jobname = jobname
        self.restart = restart
        self.work_folder_temp = f"{work_folder}/scratch"
        self.output = f"{self.work_folder_temp}/pygsm_output.txt"
        self.errlog = f"{self.work_folder_temp}/pygsm_err_msg.txt"
        self.result_file = f"{self.work_folder_temp}/pygsm_result.txt"
        self.max_gsm_iters = max_gsm_iters
        if source_path is None:
            self.source_path = "/".join(os.path.abspath(__file__).split("/")[:-3])
        else:
            self.source_path = source_path

        if not isinstance(multiplicity, int) and not isinstance(charge, int):
            info_str = "1,0"
        else:
            info_str = f"{multiplicity},{charge}"
        # Generate command
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        run_pygsm_path = os.path.join(current_file_path, "utils/run_pygsm.py")
        cmd_parts = [
            f"OMP_NUM_THREADS={nprocs} {python_exe} {run_pygsm_path}",
            f"-xyzfile {input_geo}",
            f"-calc {calc}",
            # Added Andreas
            f"-ckpt_path {ckpt_path}",
            f"-config_path {config_path}",
            f"-hessian_method {hessian_method}",
            f"-device {device}",
            # End Added Andreas
            f"-ID {jobid}",
            f"-num_nodes {num_nodes}",
            f"-nproc {nprocs}",
            f"-max_gsm_iters {max_gsm_iters}",
            f"-max_opt_steps {max_opt_steps}",
            f"-DMAX {dmax}",
            f"-ADD_NODE_TOL {add_node_tol}",
            f"-CONV_TOL {conv_tol}",
            f"-info {info_str}",
        ]

        if reactant_geom_fixed:
            cmd_parts.append("-reactant_geom_fixed")
        if product_geom_fixed:
            cmd_parts.append("-product_geom_fixed")

        self.command = " ".join(cmd_parts)
        # print("Command: \n", self.command)
        self.reason_for_failure = 0 # no failure

    def prepare_job(self):
        """Prepare GSM job by setting up working directory and input files."""
        # Create working directories if they don't exist
        os.makedirs(self.work_folder_temp, exist_ok=True)

        # Copy input geometry to scratch
        os.system(
            f"cp {self.input_geo} {self.work_folder_temp}/initial{self.jobid:03d}.xyz"
        )

        # Handle restart if needed
        if self.restart:
            opt_strings = [
                f for f in os.listdir(self.work_folder_temp) if "opt_iters_" in f
            ]
            if opt_strings:
                last_ind = max(
                    int(f.split(".xyz")[0].split("_")[-1]) for f in opt_strings
                )
                restart_string = [
                    f
                    for f in opt_strings
                    if int(f.split(".xyz")[0].split("_")[-1]) == last_ind
                ][0]
                os.system(
                    f"cp {self.work_folder_temp}/{restart_string} {self.work_folder}/restart.xyz"
                )
                self.command += f" -restart_file {self.work_folder}/restart.xyz"

        # print(f"pyGSM {self.jobname}: Finished prepping")

    def execute(self, timeout=3600):
        """Execute a GSM calculation.

        Args:
            timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.

        Returns:
            str: Status message indicating job completion or failure
        """
        if self.calculation_terminated():
            msg = f"pyGSM {self.jobname}: Finished, skipping..."
            # print(msg)
            return msg

        if os.path.isfile(self.output) and not self.restart:
            msg = f"! pyGSM {self.jobname}: Failed, skipping: {self.output}"
            print(msg)
            return msg

        try:
            os.chdir(self.work_folder)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"

            start_time = time.time()
            process = subprocess.Popen(
                self.command,
                stdout=open(self.output, "w"),
                stderr=open(self.errlog, "w"),
                shell=True,
                env=env,
                cwd=self.work_folder,
            )

            while True:
                if process.poll() is not None:
                    completed_steps = self.get_completed_steps()
                    result = subprocess.CompletedProcess(
                        args=self.command,
                        returncode=process.returncode,
                        stdout="",
                        stderr=f"Check {self.errlog}. Completed {completed_steps}/{self.max_gsm_iters} iterations",
                    )
                    break

                if time.time() - start_time > timeout:
                    process.kill()
                    completed_steps = self.get_completed_steps()
                    result = subprocess.CompletedProcess(
                        args=self.command,
                        returncode=1,
                        stdout="",
                        stderr=f"pyGSM {self.jobname}: Timed out after {time.time() - start_time:.1f}s > {timeout}s (completed {completed_steps}/{self.max_gsm_iters} iterations)",
                    )
                    break

                time.sleep(1)

            execution_time = time.time() - start_time

            if result.returncode == 0:
                msg = f"pyGSM {self.jobname}: Finished in {execution_time:.1f}s (completed {completed_steps}/{self.max_gsm_iters} iterations)"
            else:
                msg = f"pyGSM {self.jobname}: Failed. Returncode: {result.returncode}. Check {self.errlog}"
                msg += f"\nresult: \n{result}"

            # copy the last 50 lines of the output file to a new result file
            with open(self.output, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(self.result_file, "a", encoding="utf-8") as f:
                f.write("".join(lines[-50:]))
                f.write("\n" + msg)

            return msg

        finally:
            if process.poll() is None:
                process.kill()

    def calculation_terminated(self) -> bool:
        """Check if the calculation has terminated.

        Returns:
            bool: True if calculation terminated, False otherwise
        """
        if not os.path.isfile(self.output):
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if any(
                    marker in line
                    for marker in ["Printing string to", "Finished GSM!", "error"]
                ):
                    return True
        return False

    def calculation_terminated_successfully(self) -> bool:
        """Check if the calculation terminated successfully.

        Returns:
            bool: True if calculation completed successfully, False otherwise
        """
        if not os.path.isfile(self.output):
            self.reason_for_failure = 404 # output file not found
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if "Finished GSM!" in line:
                    return True
        
        self.reason_for_failure = 100 # unknown reason
        return False

    def read_error_content(self) -> str:
        """Read error content from the error log file.

        Returns:
            str: Error content if file exists and has content, empty string otherwise
        """
        if not os.path.isfile(self.errlog):
            return ""

        try:
            with open(self.errlog, "r", encoding="utf-8") as f:
                content = f.read().strip()
                content = "# " + self.errlog + "\n" + content
                content += "# ---"
                return content
        except Exception:
            return ""

    def get_completed_steps(self) -> int:
        """Count the number of completed GSM iteration steps by parsing output file.

        Returns:
            int: Number of completed GSM iterations based on V_profile entries
        """
        if not os.path.isfile(self.output):
            return 0

        try:
            with open(self.output, "r", encoding="utf-8") as f:
                lines = f.readlines()

            step_count = 0
            for line in lines:
                if "V_profile:" in line:
                    step_count += 1

            return step_count
        except Exception:
            return 0

    def find_correct_TS(self, tight=True) -> int:
        """Find the transition state node index.
        Inputs:
            Tight: Only if only one peak is found, return this geometry as TS.
            Comment Andreas: does not work well. Recommended to set to False!

        Returns:
            int: Index of transition state node, or False if not found
        """
        if not self.calculation_terminated_successfully():
            return False

        with open(self.output, "r", encoding="utf-8") as f:
            lines = f.readlines()

        energies = []
        for line in reversed(lines):
            if "V_profile:" in line:
                energies = [float(i) for i in line.split()[1:]]
                break

        if len(energies) < 5:
            print(f"! find_correct_TS there are less than 5 energies in {self.output}")
            return False

        if max(energies) > 1000:
            print(f"! find_correct_TS the max energy is greater than 1000 in {self.output}")
            return False

        peaks = []
        # Check endpoints
        if energies[1] > max(energies[0], energies[2], energies[3]):
            peaks.append(1)
        if energies[-2] > max(energies[-1], energies[-3], energies[-4]):
            peaks.append(len(energies) - 2)

        # Check internal points
        for i in range(2, len(energies) - 2):
            if energies[i] > max(energies[i - 1], energies[i - 2]) and energies[
                i
            ] > max(energies[i + 1], energies[i + 2]):
                peaks.append(i)

        # check and return peak
        if not peaks:
            print(f"! find_correct_TS there are no peaks in {energies} {self.output}")
            return False

        if len(peaks) == 1:
            return peaks[0]
        else:
            if tight:
                print(f"! find_correct_TS there are multiple peaks in {energies} {self.output}")
                return False
            else:
                # Find the peak with the maximum energy
                return max(peaks, key=lambda ind: energies[ind])

    def get_strings(self):
        """Get the final optimized string of images.

        Returns:
            list: List of optimized geometries if successful, False otherwise
        """
        strings_xyz = f"{self.work_folder}/opt_converged_{self.jobid:03d}.xyz"
        if os.path.exists(strings_xyz):
            return xyz_parse(strings_xyz, multiple=True)
        else:
            strings_xyz = [
                os.path.join(self.work_folder, i)
                for i in os.listdir(f"{self.work_folder}")
                if "opt_converged_" in i
            ][0]
            return xyz_parse(strings_xyz, multiple=True)

    def get_TS(self, tight=True):
        """Get the transition state geometry.
        Inputs:
            Tight: Only if only one peak is found, return this geometry as TS

        Returns:
            tuple: (elements, coordinates) if successful, (False, []) otherwise
        """
        if not self.calculation_terminated_successfully():
            return False, []

        ts_index = self.find_correct_TS(tight=tight)
        if not ts_index:
            return False, []

        images = self.get_strings()
        if not images:
            return False, []

        return images[ts_index]
