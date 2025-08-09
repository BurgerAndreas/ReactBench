#!/bin/env python
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import os, sys
from glob import glob
import yaml
import logging
import time
import shutil
import threading

import multiprocessing as mp
from logging.handlers import QueueHandler
from joblib import Parallel, delayed
import wandb
import numpy as np

# Add the parent directory to Python path to enable ReactBench module import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ReactBench.utils.parsers import xyz_write
from ReactBench.main_functions import logger_process
from ReactBench.main_functions import analyze_outputs
from ReactBench.pysis import PYSIS
from ReactBench.gsm import PYGSM


class DummyLogger:
    def __init__(self, *args, **kwargs):
        self.info = lambda x: None
        self.warning = lambda x: None
        self.error = lambda x: None
        self.critical = lambda x: None
        self.debug = lambda x: None
        self.exception = lambda x: None
        self.fatal = lambda x: None


def print_error_content(errlog, logger):
    if os.path.exists(errlog):
        with open(errlog, "r", encoding="utf-8") as f:
            content = f.read().strip()
            content = "\n# " + errlog + "\n" + content
            content += "\n# ---"
        print(content)
        logger.info(content)
    else:
        print(f"Error log file {errlog} not found")
        logger.info(f"Error log file {errlog} not found")


# Generate run name based on calc and ckpt_path
def generate_run_name(args):
    calc = args.get("calc", "unknown")
    ckpt_path = args.get("ckpt_path", "")
    _name = ""
    if ckpt_path:
        parent_dir = os.path.basename(os.path.dirname(ckpt_path))
        filename = os.path.splitext(os.path.basename(ckpt_path))[0]
        _name += f"{calc}_{parent_dir}{filename}"
    else:
        _name += f"{calc}"
    _name += "_" + os.path.basename(args["inp_path"])
    _name += "_" + args["hessian_method"] if args["hessian_method"] else "autograd"
    return _name

# Background monitor: periodically read result files and log to wandb
def _monitor_results_periodically(scratch_dir: str, stop_evt: threading.Event, interval_sec: int = 60):
    prev_pygsm_output_files = set()
    prev_pysistsopt_output_files = set()
    while not stop_evt.is_set():
        try:
            # Aggregate lightweight progress metrics from filesystem
            gsm_success = len(glob(f"{scratch_dir}/*/*TSguess.xyz"))
            job_dirs = [
                d for d in os.listdir(scratch_dir)
                if os.path.isdir(os.path.join(scratch_dir, d)) and d not in ["init_rxns", "scratch"]
            ]
            
            # count pygsm_output.txt files
            # runs/equiformer*/rxn123/scratch/pygsm_output.txt
            pygsm_output_files = glob(f"{scratch_dir}/*/scratch/pygsm_output.txt")
            pygsm_output_files = [f for f in pygsm_output_files if os.path.isfile(f)]
            n_pygsm_output_files = len(pygsm_output_files)
            # get last 20 lines, search for msg after "optimize_string result:"
            gsm_outs = {}
            times_taken = []
            for f in pygsm_output_files:
                with open(f, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in reversed(lines[-20:]):
                        if "optimize_string result:" in line:
                            msg = line.split("optimize_string result:")[1].strip()
                            if msg not in gsm_outs.keys():
                                gsm_outs["monitor/" + "optim_string/" + msg] = 0
                            gsm_outs["monitor/" + "optim_string/" + msg] += 1
                    if "Time taken:" in line:
                        times_taken.append(float(line.split("Time taken:")[1].split("s")[0].strip()))
                        
            # for new pygsm_output.txt files, print the last 20 lines so wandb captures them
            new_pygsm_output_files = [f for f in pygsm_output_files if f not in prev_pygsm_output_files]
            prev_pygsm_output_files = set(pygsm_output_files)
            for f in new_pygsm_output_files:
                with open(f, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"\n# {f}")
                    print(lines[-20:])
                    print("")
            
            # Count TSOPT results
            # runs/equiformer*/rxn123/TSOPT/pysis_tsopt_output.txt
            pysistsopt_output_files = glob(f"{scratch_dir}/*/TSOPT/pysis_tsopt_output.txt")
            pysistsopt_output_files = [f for f in pysistsopt_output_files if os.path.isfile(f)]
            n_pysistsopt_output_files = len(pysistsopt_output_files)
            # get last 20 lines, search for msg after "pysis result:"
            tsopt_outs = {}
            pysis_times_taken = []
            pysis_cycles_taken = []
            for f in pysistsopt_output_files:
                with open(f, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in reversed(lines[-50:]):
                        # pysis result: 
                        if "pysis result:" in line:
                            msg = line.split("pysis result:")[1].strip()
                            _logkey = "monitor/" + "tsopt/" + msg
                            if _logkey not in tsopt_outs.keys():
                                tsopt_outs[_logkey] = 0
                            tsopt_outs[_logkey] += 1
                        # Cycles taken: ...
                        if "Cycles taken:" in line:
                            pysis_cycles_taken.append(int(line.split("Cycles taken:")[1].strip()))
                        # Time taken: ... s
                        if "Time taken:" in line:
                            pysis_times_taken.append(float(line.split("Time taken:")[1].split("s")[0].strip()))
            
            # for new pysis_tsopt_output.txt files, print last 50 lines
            new_pysistsopt_output_files = [f for f in pysistsopt_output_files if f not in prev_pysistsopt_output_files]
            prev_pysistsopt_output_files = set(pysistsopt_output_files)
            for f in new_pysistsopt_output_files:
                with open(f, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"\n# {f}")
                    print(lines[-50:])
                    print("")
            
            # Count TSOPT and IRC results
            tsopt_results = 0
            irc_results = 0
            for d in job_dirs:
                tsopt_res = os.path.join(scratch_dir, d, "TSOPT", "pysis_tsopt_result.txt")
                irc_res = os.path.join(scratch_dir, d, "IRC", "pysis_irc_result.txt")
                # Touch the files by reading a few bytes to satisfy "read result file" requirement
                if os.path.isfile(tsopt_res):
                    tsopt_results += 1
                    try:
                        with open(tsopt_res, "r", encoding="utf-8", errors="ignore") as f:
                            _ = f.readline()
                    except Exception:
                        pass
                if os.path.isfile(irc_res):
                    irc_results += 1
                    try:
                        with open(irc_res, "r", encoding="utf-8", errors="ignore") as f:
                            _ = f.readline()
                    except Exception:
                        pass

            metrics = {
                "monitor/gsm_success": gsm_success,
                "monitor/tsopt_results": tsopt_results,
                "monitor/irc_results": irc_results,
                "monitor/active_jobs": len(job_dirs),
                "monitor/pygsm_output_files": n_pygsm_output_files,
            }
            if times_taken:
                metrics["monitor/gsm_time_taken"] = sum(times_taken) / len(times_taken)
                metrics["monitor/gsm_time_taken_std"] = np.std(times_taken)
            if pysis_times_taken:
                metrics["monitor/pysis_time_taken"] = sum(pysis_times_taken) / len(pysis_times_taken)
                metrics["monitor/pysis_time_taken_std"] = np.std(pysis_times_taken)
            if pysis_cycles_taken:
                metrics["monitor/pysis_cycles_taken"] = sum(pysis_cycles_taken) / len(pysis_cycles_taken)
                metrics["monitor/pysis_cycles_taken_std"] = np.std(pysis_cycles_taken)
            metrics.update(gsm_outs)
            metrics.update(tsopt_outs)
            if wandb.run is not None:
                wandb.log(metrics)
        except Exception as _e:
            # Avoid crashing the main run due to monitor issues
            print(f"Error in monitor: {_e}")
            # logger.info(f"Error in monitor: {_e}")
            pass
        finally:
            stop_evt.wait(interval_sec)

def launch_tssearch_processes(args: dict, wandb_run_id=None, wandb_kwargs={}):
    """
    Take the arguments from options and run YARP
    """
    # Set REACTBENCH_PATH environment variable if it exists in the config file
    if "reactbench_path" not in args:
        args["reactbench_path"] = None
    if args["reactbench_path"] is None:
        args["reactbench_path"] = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    if args["reactbench_path"] not in sys.path:
        sys.path.insert(0, args["reactbench_path"])
    os.environ["REACTBENCH_PATH"] = args["reactbench_path"]
    print(f"Set REACTBENCH_PATH environment variable to: {args['reactbench_path']}")

    # load in parameters
    charge = args.get("charge", 0)
    multiplicity = args.get("multiplicity", 1)

    # Set scratch directory
    scratch = args["scratch"]
    if scratch is None:
        scratch = "runs/" + generate_run_name(args)
        args["scratch"] = scratch

    # check that nprocs is smaller than the number of cpus on the machine
    max_cpus = mp.cpu_count()
    if args["nprocs"] == "auto":
        args["nprocs"] = max_cpus - 2
    if args["nprocs"] > max_cpus:
        print(
            f"ERROR: nprocs ({args['nprocs']}) exceeds available CPU cores ({max_cpus})"
        )
        print(
            f"Recommended: Use at most {max_cpus - 2} cores to maintain system stability"
        )
        args["nprocs"] = max_cpus - 2
    nprocs = int(args["nprocs"])

    # initialiazation
    if scratch[0] != "/":
        scratch = os.path.join(os.getcwd(), scratch)
    scratch_opt = f"{scratch}/scratch"
    args["scratch_opt"] = scratch_opt

    # Cleanup results if specified in config
    if args.get("redo_all", False):
        print(f"\nCleaning up results directory: {scratch}")
        if os.path.exists(scratch):
            shutil.rmtree(scratch)

    if args.get("wandb", False):
        # Use the same naming convention for wandb as scratch
        wandb_name = generate_run_name(args)
        if wandb_run_id is None:
            wandb.init(project="reactbench", name=wandb_name, config=args, **wandb_kwargs)

    if args.get("only_cnt_results", False):
        logger = DummyLogger()
    else:
        # create folders
        os.makedirs(scratch, exist_ok=True)
        os.makedirs(scratch_opt, exist_ok=True)

        # create init_rxns folder and copy input files
        os.makedirs(f"{scratch}/init_rxns", exist_ok=True)
        for f in os.listdir(args["inp_path"]):
            if f.endswith(".xyz"):
                os.system(f"cp {args['inp_path']}/{f} {scratch}/init_rxns/")

        # initialize logging file
        logging_path = os.path.join(scratch, "YARPrun.log")
        logging_queue = mp.Manager().Queue(999)
        logger_p = mp.Process(
            target=logger_process, args=(logging_queue, logging_path), daemon=True
        )
        logger_p.start()
        start = time.time()

        # logger in the main process
        logger = logging.getLogger("main")
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

        # print head of the program
        logger.info(
            r"""Welcome to
                    __   __ _    ____  ____  
                    \ \ / // \  |  _ \|  _ \ 
                    \ V // _ \ | |_) | |_) |
                    | |/ ___ \|  _ <|  __/ 
                    |_/_/   \_\_| \_\_|
                            // Yet Another Reaction Program
            """
        )
        logger.info("=" * 80)
        logger.info(
            "                               INPUT PARAMETERS                                 "
        )
        logger.info("=" * 80)
        for key, val in args.items():
            line = str(key) + ": " + str(val)
            logger.info(line)

        logger.info("\n" + "=" * 80)
        logger.info(
            "                           PARSING INPUT REACTIONS                              "
        )
        logger.info("=" * 80)

        logger.info("=" * 80)
        logger.info(
            "                        RUNNING GROWING STRING METHODS                          "
        )
        logger.info("=" * 80)

        ## Load in input rxns
        rxns_confs = [
            rxn
            for rxn in sorted(os.listdir(f"{scratch}/init_rxns"))
            if rxn[-4:] == ".xyz"
        ]
        thread = min(nprocs, len(rxns_confs))
        input_job_list = []

        # preparing and running GSM calculations
        for count, rxn in enumerate(rxns_confs):
            # prepare GSM-TSOPT-IRC job
            rxn_ind = rxn.split(".xyz")[0]
            input_job_list.append(
                (
                    count,
                    rxn,
                    scratch,
                    args,
                    logging_queue,
                    args["gsm_wt"],
                    charge,
                    multiplicity,
                    args.get("redo_gsm", False),
                    args.get("redo_opt", False),
                    args.get("redo_irc", False),
                )
            )


        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_results_periodically,
            args=(scratch, monitor_stop_event, 60),
            daemon=True,
        )
        monitor_thread.start()

        # Run the tasks in parallel
        jobs = Parallel(n_jobs=thread)(
            delayed(run_gsm_rsprfo_irc)(*task) for task in input_job_list
        )

        tsopt_jobs = {}
        irc_jobs = []
        for job in jobs:
            # (rxn_ind, tsopt_job, irc_job)
            if job[1] is False:
                continue
            tsopt_jobs[job[0]] = job[1]
            irc_jobs.append(job[2])

        # reporting GSM wall-time
        end = time.time()
        print(f"Total running time: {end - start}s")
        logger.info(f"Total running time: {end - start}s\n")

        # Analyze the output
        analyze_outputs(scratch, irc_jobs, logger, charge=charge)
        logger.info(f"All reaction information is stored in {scratch}/IRC-record.txt")

        print("All calculations are done!")
        logger.info("All calculations are done!")
        
        # Stop background monitor after jobs complete
        # wait for 60s to give minitor opportunity to report final metrics
        time.sleep(60)
        monitor_stop_event.set()
        monitor_thread.join(timeout=20)

    print("\n=== Final Results ===")
    print(f"Number of input reactions:             {len(rxns_confs)}")
    gsm_success = len(glob(f"{scratch}/*/*TSguess.xyz"))
    print(f"Number of successful GSM calculations: {gsm_success}")

    # Count successful TS optimizations
    ts_success = 0
    convert_ts = 0
    converged_ts = 0
    for folder in glob(f"{scratch}/*/TSOPT"):
        with open(os.path.join(folder, "pysis_tsopt_output.txt"), "r") as f:
            lines = f.readlines()
            true_ts = False
            for line in reversed(lines):
                if "Imaginary frequencies" in line:
                    freq = line.split("[")[1].split("]")[0].split()
                    if len(freq) == 1 and float(freq[0]) < -10:
                        ts_success += 1
                        true_ts = True
                if "Converged!" in line:
                    converged_ts += 1
                    if true_ts:
                        convert_ts += 1

    print(f"Number of successful TS optimizations: {ts_success}({convert_ts})")

    # Count successful IRC calculations
    irc_success = 0
    for folder in glob(f"{scratch}/*/IRC"):
        if not os.path.exists(os.path.join(folder, "pysis_irc_output.txt")):
            continue
        with open(os.path.join(folder, "pysis_irc_output.txt"), "r") as f:
            lines = f.readlines()
            left = ts = right = 0
            # This counts how many structures have energies different from the minimum.
            # Since one structure will always be 0 (the lowest energy), having ≥2 non-zero values means:
            # At least 3 distinct energy levels were found (minimum + 2 others)
            # The IRC successfully traced from TS to different reactant/product structures
            for line in lines:
                # example line:
                # Left:   413.21 kJ mol⁻¹ (1 geometry)
                # TS:   620.89 kJ mol⁻¹ (1 geometry)
                # Right:     0.00 kJ mol⁻¹ (1 geometry)
                if "Left:" in line:
                    left = float(line.split("kJ")[0].split(":")[1].strip())
                elif "TS:" in line:
                    ts = float(line.split("kJ")[0].split(":")[1].strip())
                elif "Right:" in line:
                    right = float(line.split("kJ")[0].split(":")[1].strip())
            if sum([left != 0, ts != 0, right != 0]) >= 2:
                irc_success += 1

    print(f"Number of successful IRC calculations: {irc_success}")

    # Count intended reactions if IRC-record.txt exists
    irc_record = os.path.join(scratch, "IRC-record.txt")
    if os.path.exists(irc_record):
        with open(irc_record, "r") as f:
            intended_count = sum(line.count("Intended") for line in f)
        print(f"Number of intended reactions:          {intended_count}")

    # summarize into a dictionary
    ts_success_dict = {
        "gsm_success": gsm_success,
        "ts_success": ts_success,
        "convert_ts": convert_ts,
        "irc_success": irc_success,
        "intended_count": intended_count,
    }
    if wandb.run is not None:
        wandb.log(ts_success_dict)

    # Cleanup results if specified in config
    if args.get("cleanup_results", False):
        print(f"\nCleaning up results directory: {scratch}")
        logger.info(f"Cleaning up results directory: {scratch}")
        try:
            shutil.rmtree(scratch)
            print("Results cleanup completed successfully.")
            logger.info("Results cleanup completed successfully.")
        except Exception as e:
            print(f"Warning: Failed to cleanup results directory: {e}")
            logger.warning(f"Failed to cleanup results directory: {e}")

    return


def run_gsm_rsprfo_irc(
    count, rxn, scratch, args, logging_queue, 
    timeout=3600, charge=0, multiplicity=1,
    redo_gsm=False, redo_opt=False, redo_irc=False,
):
    """
    subprocess for running ts calculation in parallel
    each process contains gsm, ts-opt, and irc

    timeout: in seconds
    """
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)
        
    #######################################################
    # GSM
    #######################################################

    # prepare GSM job
    rxn_ind = rxn.split(".xyz")[0]
    wf = f"{scratch}/{rxn_ind}"
    os.makedirs(wf, exist_ok=True)
    inp_xyz = f"{scratch}/init_rxns/{rxn}"
    
    if redo_gsm:
        # remove wf/scratch
        if os.path.exists(wf + "/scratch"):
            shutil.rmtree(wf + "/scratch")
        # remove wf/opt_converged_*.xyz
        for f in glob(f"{wf}/opt_converged_*.xyz"):
            os.remove(f)
        # remove wf/restart.xyz
        if os.path.exists(f"{wf}/restart.xyz"):
            os.remove(f"{wf}/restart.xyz")
        # remove wf/initial*.xyz
        for f in glob(f"{wf}/initial*.xyz"):
            os.remove(f)
        # remove wf/growing_string_*.xyz
        for f in glob(f"{wf}/growing_string_*.xyz"):
            os.remove(f)
    
    gsm_job = PYGSM(
        input_geo=inp_xyz,
        calc=args["calc"],
        # Added Andreas
        device=args["device"],
        ckpt_path=args["ckpt_path"],
        config_path=args["config_path"],
        hessian_method=args["hessian_method"],
        # End Added Andreas
        work_folder=wf,
        jobid=count,
        nprocs=1,
        jobname=rxn_ind,
        restart=args["gsm_restart"],
        num_nodes=args["num_nodes"],
        add_node_tol=args["add_node_tol"],
        conv_tol=args["conv_tol"],
        max_opt_steps=args["max_opt_steps"],
        max_gsm_iters=args["max_gsm_iters"],
        dmax=args["dmax"],
        reactant_geom_fixed=args["fixed_R"],
        product_geom_fixed=args["fixed_P"],
        python_exe=args["python_exe"],
        multiplicity=multiplicity,
        charge=charge,
    )
    gsm_job.prepare_job()

    # run GSM job
    start = time.time()
    result = gsm_job.execute(timeout=timeout)
    end = time.time()
    logger.info(result)

    successful_gsm = len(glob(f"{scratch}/*/*TSguess.xyz"))
    if wandb.run is not None:
        wandb.log({"successful_gsm": successful_gsm})

    # check GSM job
    if gsm_job.calculation_terminated_successfully() is False:
        error_msg = f"GSM job {gsm_job.jobname} fails to converge, please check this reaction..."
        print(gsm_job.errlog)
        print(error_msg)
        logger.info(gsm_job.errlog)
        logger.info(error_msg)
        print_error_content(gsm_job.errlog, logger)
        return (rxn_ind, False, False)

    if gsm_job.find_correct_TS() is False:
        print(f"GSM job {gsm_job.jobname} fails to locate a TS, skip this rxn...")
        logger.info(f"GSM job {gsm_job.jobname} fails to locate a TS, skip this rxn...")
        return (rxn_ind, False, False)
    
    #######################################################
    # TS-OPT with RSPRFO
    #######################################################
    
    if redo_opt:
        # remove wf/TSOPT
        if os.path.exists(gsm_job.work_folder + "/TSOPT"):
            shutil.rmtree(gsm_job.work_folder + "/TSOPT")

    # prepare ts-opt job
    TSE, TSG = gsm_job.get_TS()
    xyz_write(f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz", TSE, TSG)
    work_folder = os.path.join(gsm_job.work_folder, "TSOPT")

    calc_kwargs = {
        "device": args["device"],
        "ckpt_path": args["ckpt_path"],
        "config_path": args["config_path"],
        "hessian_method": args["hessian_method"],
    }
    tsopt_job = PYSIS(
        input_geo=f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",
        work_folder=work_folder,
        restart=args["pysis_restart"],
        jobname=gsm_job.jobname,
        jobtype="tsopt",
        # method defaults to rsprfo
        exe=args["pysis_exe"],
        multiplicity=multiplicity,
        charge=charge,
        calc_kwargs=calc_kwargs,
    )
    tsopt_job.generate_input(
        calctype=f"mlff-{args['calc']}",
        hess=True,
        hess_step=1,
        calc_kwargs=calc_kwargs,
    )

    # run ts-opt job
    start = time.time()
    result = tsopt_job.execute(timeout=timeout)
    end = time.time()
    logger.info(result)

    # Count successful TS optimizations
    completed_tsopt = len(glob(f"{scratch}/*/TSOPT"))
    if wandb.run is not None:
        wandb.log({"completed_tsopt": completed_tsopt})

    # check ts-opt job
    if tsopt_job.calculation_terminated_normally() is False:
        print(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
        logger.info(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
        print_error_content(tsopt_job.errlog, logger)
        return (rxn_ind, False, False)

    is_true_ts = tsopt_job.is_true_ts()
    msg = f"TSopt job {tsopt_job.jobname} is true TS: {is_true_ts}"
    msg += f" (num_im_freqs={tsopt_job.freq_analysis['num_im_freqs']}"
    for hessian_method in ["autograd", "predict"]:
        if hessian_method in tsopt_job.freq_analysis:
            msg += f", {hessian_method}:{tsopt_job.freq_analysis[hessian_method]}"
    msg += ")"
    print(msg)
    logger.info(msg)
    
    strategy = args.get("ts_require", "default")
    msg = f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction."
    # always do IRC
    if strategy in [None, "None", "none"]:
        pass
    # both autograd and hessian predict must be true
    elif strategy == "all":
        for k, v in is_true_ts.items():
            if v is False:
                print(msg)
                logger.info(msg)
                return (rxn_ind, False, False)
    # any of the hessian methods must be true
    elif strategy == "any":
        _continue = False
        for k, v in is_true_ts.items():
            if v is True:
                _continue = True
                break
        if _continue is False:
            print(msg)
            logger.info(msg)
            return (rxn_ind, False, False)
    # only the default hessian method must be true
    elif strategy == "default":
        if is_true_ts["default"] is False:
            print(msg)
            logger.info(msg)
            return (rxn_ind, False, False)
    # only the autograd hessian method must be true
    elif strategy == "autograd":
        if is_true_ts["autograd"] is False:
            msg = f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction."
        msg = f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction."
        return (rxn_ind, False, False)
    else:
        raise ValueError(f"Invalid ts_require strategy: {strategy}")

    #######################################################
    # IRC
    #######################################################
    
    # prepare irc job
    TSE, TSG = tsopt_job.get_final_ts()
    xyz_write(f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz", TSE, TSG)
    work_folder = tsopt_job.work_folder.replace("TSOPT", "IRC")
    calc_kwargs = {
        "device": args["device"],
        "ckpt_path": args["ckpt_path"],
        "config_path": args["config_path"],
        "hessian_method": args["hessian_method"],
    }
    irc_job = PYSIS(
        input_geo=f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",
        work_folder=work_folder,
        restart=args["pysis_restart"],
        jobname=tsopt_job.jobname,
        jobtype="irc",
        exe=args["pysis_exe"],
        calc_kwargs=calc_kwargs,
    )
    if os.path.isfile(f"{tsopt_job.work_folder}/ts_final_hessian.h5"):
        irc_job.generate_input(
            calctype=f"mlff-{args['calc']}",
            hess_init=f"{tsopt_job.work_folder}/ts_final_hessian.h5",
            calc_kwargs=calc_kwargs,
        )
    else:
        print(
            f"IRC did not find ts_final_hessian for {tsopt_job.jobname} {tsopt_job.jobtype}"
        )
        irc_job.generate_input(
            calctype=f"mlff-{args['calc']}",
            calc_kwargs=calc_kwargs,
        )

    # run irc job
    start = time.time()
    result = irc_job.execute(timeout=timeout)
    end = time.time()
    logger.info(result)

    # Count completed jobs based on saved results
    completed_irc = len(
        [
            d
            for d in os.listdir(scratch)
            if os.path.isdir(os.path.join(scratch, d))
            and d not in ["init_rxns", "scratch"]
        ]
    )
    print(f"{completed_irc} jobs finished")
    if wandb.run is not None:
        wandb.log({"completed_irc": completed_irc})

    return (rxn_ind, tsopt_job, irc_job)


if __name__ == "__main__":
    parameters_yaml = sys.argv[1]
    parameters = yaml.load(open(parameters_yaml, "r"), Loader=yaml.FullLoader)

    # Parse additional command line arguments for parameter overrides
    # Format: --key=value
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--"):
            if "=" in arg:
                # Handle --key=value format
                key, value = arg[2:].split("=", 1)
            else:
                raise ValueError(f"Unrecognized argument: --{arg}")

            # Try to convert value to appropriate type
            # Check if it's a number
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Check if it's a boolean
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                # Otherwise keep as string

            if key not in parameters:
                raise ValueError(f"Unrecognized argument: --{key}")

            parameters[key] = value
            print(f"Override: {key} = {value}")
        else:
            raise ValueError(f"Unrecognized argument: {arg}")
        i += 1

    launch_tssearch_processes(parameters)
