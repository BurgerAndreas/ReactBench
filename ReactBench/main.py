#!/bin/env python
# Author: Qiyuan Zhao (zhaoqy1996@gmail.com)

import os, sys
from glob import glob
import yaml
import logging
import time
import pyjokes
import shutil

import multiprocessing as mp
from logging.handlers import QueueHandler
from joblib import Parallel, delayed

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


def main(args: dict):
    """
    Take the arguments from options and run YARP
    """
    # Set REACTBENCH_PATH environment variable if it exists in the config file
    if "reactbench_path" in args:
        os.environ["REACTBENCH_PATH"] = args["reactbench_path"]
        print(f"Set REACTBENCH_PATH environment variable to: {args['reactbench_path']}")

    # load in parameters
    scratch = args["scratch"]
    nprocs = int(args["nprocs"])
    charge = args.get("charge", 0)
    multiplicity = args.get("multiplicity", 1)

    # initialiazation
    if scratch[0] != "/":
        scratch = os.path.join(os.getcwd(), scratch)
    scratch_opt = f"{scratch}/scratch"
    args["scratch_opt"] = scratch_opt

    # create folders
    if os.path.isdir(scratch) is False:
        os.mkdir(scratch)
    if os.path.isdir(scratch_opt) is False:
        os.mkdir(scratch_opt)

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
    logger.info(
        "================================================================================"
    )
    logger.info(
        "                               INPUT PARAMETERS                                 "
    )
    logger.info(
        "================================================================================"
    )
    for key, val in args.items():
        line = str(key) + ": " + str(val)
        logger.info(line)

    logger.info(
        "\n================================================================================"
    )
    logger.info(
        "                           PARSING INPUT REACTIONS                              "
    )
    logger.info(
        "================================================================================"
    )
    logger.info("\N{THINKING FACE} " + pyjokes.get_joke())

    logger.info(
        "================================================================================"
    )
    logger.info(
        "                        RUNNING GROWING STRING METHODS                          "
    )
    logger.info(
        "================================================================================"
    )
    logger.info("\N{NERD FACE} " + pyjokes.get_joke())

    ## Load in input rxns
    rxns_confs = [
        rxn for rxn in sorted(os.listdir(f"{scratch}/init_rxns")) if rxn[-4:] == ".xyz"
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
            )
        )

    # Run the tasks in parallel
    jobs = Parallel(n_jobs=thread)(delayed(ts_calc)(*task) for task in input_job_list)
    tsopt_jobs = {}
    irc_jobs = []
    for job in jobs:
        if job[1] is False:
            continue
        tsopt_jobs[job[0]] = job[1]
        irc_jobs.append(job[2])

    # reporting GSM wall-time
    end = time.time()
    print(f"Total running time: {end-start}s")
    logger.info(f"Total running time: {end-start}s\n")

    # Analyze the output
    analyze_outputs(scratch, irc_jobs, logger, charge=charge)
    logger.info(f"All reaction information is stored in {scratch}/IRC-record.txt")

    print("All calculations are done!")
    logger.info(
        "All calculations are done!"
    )

    print("\n=== Final Results ===")
    print(f"Number of input reactions:             {len(rxns_confs)}")
    print(
        f"Number of successful GSM calculations: {len(glob(f'{scratch}/*/*TSguess.xyz'))}"
    )

    # Count successful TS optimizations
    ts_success = 0
    convert_ts = 0
    for folder in glob(f"{scratch}/*/TSOPT"):
        with open(os.path.join(folder, "output.txt"), "r") as f:
            lines = f.readlines()
            true_ts = False
            for line in reversed(lines):
                if "Imaginary frequencies" in line:
                    freq = line.split("[")[1].split("]")[0].split()
                    if len(freq) == 1 and float(freq[0]) < -10:
                        ts_success += 1
                        true_ts = True
                if "Converged!" in line and true_ts:
                    convert_ts += 1

    print(f"Number of successful TS optimizations: {ts_success}({convert_ts})")

    # Count successful IRC calculations
    irc_success = 0
    for folder in glob(f"{scratch}/*/IRC"):
        if not os.path.exists(os.path.join(folder, "output.txt")):
            continue
        with open(os.path.join(folder, "output.txt"), "r") as f:
            lines = f.readlines()
            left = ts = right = 0
            for line in lines:
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


def ts_calc(
    count, rxn, scratch, args, logging_queue, timeout=3600, charge=0, multiplicity=1
):
    """
    subprocess for running ts calculation in parallel
    each process contains gsm, ts-opt, and irc
    """
    # set up logger
    logger = logging.getLogger("main")
    # Add handler only if it doesn't already exist
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(logging_queue))
        logger.setLevel(logging.INFO)

    # prepare GSM job
    rxn_ind = rxn.split(".xyz")[0]
    wf = f"{scratch}/{rxn_ind}"
    if os.path.isdir(wf) is False:
        os.mkdir(wf)
    inp_xyz = f"{scratch}/init_rxns/{rxn}"
    gsm_job = PYGSM(
        input_geo=inp_xyz,
        calc=args["calc"],
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

    # check GSM job
    if gsm_job.calculation_terminated_successfully() is False:
        print(
            f"GSM job {gsm_job.jobname} fails to converge, please check this reaction..."
        )
        logger.info(
            f"GSM job {gsm_job.jobname} fails to converge, please check this reaction..."
        )
        return (rxn_ind, False, False)

    if gsm_job.find_correct_TS() is False:
        print(f"GSM job {gsm_job.jobname} fails to locate a TS, skip this rxn...")
        logger.info(f"GSM job {gsm_job.jobname} fails to locate a TS, skip this rxn...")
        return (rxn_ind, False, False)

    # prepare ts-opt job
    TSE, TSG = gsm_job.get_TS()
    xyz_write(f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz", TSE, TSG)
    work_folder = os.path.join(gsm_job.work_folder, "TSOPT")

    tsopt_job = PYSIS(
        input_geo=f"{gsm_job.work_folder}/{gsm_job.jobname}-TSguess.xyz",
        work_folder=work_folder,
        restart=args["pysis_restart"],
        jobname=gsm_job.jobname,
        jobtype="tsopt",
        exe=args["pysis_exe"],
        multiplicity=multiplicity,
        charge=charge,
    )
    tsopt_job.generate_input(calctype=f'mlff-{args["calc"]}', hess=True, hess_step=1)

    # run ts-opt job
    start = time.time()
    result = tsopt_job.execute(timeout=timeout)
    end = time.time()
    logger.info(result)

    # check ts-opt job
    if tsopt_job.calculation_terminated_normally() is False:
        print(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
        logger.info(f"TSopt job {tsopt_job.jobname} fails, skip this reaction...")
        return (rxn_ind, False, False)

    if tsopt_job.is_true_ts() is False:
        print(
            f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction..."
        )
        logger.info(
            f"TSopt job {tsopt_job.jobname} fails to locate a true transition state, skip this reaction..."
        )
        return (rxn_ind, False, False)

    # prepare irc job
    TSE, TSG = tsopt_job.get_final_ts()
    xyz_write(f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz", TSE, TSG)
    work_folder = tsopt_job.work_folder.replace("TSOPT", "IRC")
    irc_job = PYSIS(
        input_geo=f"{tsopt_job.work_folder}/{tsopt_job.jobname}-TS.xyz",
        work_folder=work_folder,
        restart=args["pysis_restart"],
        jobname=tsopt_job.jobname,
        jobtype="irc",
        exe=args["pysis_exe"],
    )
    if os.path.isfile(f"{tsopt_job.work_folder}/ts_final_hessian.h5"):
        irc_job.generate_input(
            calctype=f'mlff-{args["calc"]}',
            hess_init=f"{tsopt_job.work_folder}/ts_final_hessian.h5",
        )
    else:
        irc_job.generate_input(calctype=f'mlff-{args["calc"]}')

    # run irc job
    start = time.time()
    result = irc_job.execute(timeout=timeout)
    end = time.time()
    logger.info(result)

    return (rxn_ind, tsopt_job, irc_job)


if __name__ == "__main__":
    parameters_yaml = sys.argv[1]
    parameters = yaml.load(open(parameters_yaml, "r"), Loader=yaml.FullLoader)

    # Parse additional command line arguments for parameter overrides
    # Format: --key=value or --key value
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--"):
            if "=" in arg:
                # Handle --key=value format
                key, value = arg[2:].split("=", 1)
            else:
                # Handle --key value format
                key = arg[2:]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                    i += 1
                    value = sys.argv[i]
                else:
                    print(f"Warning: No value provided for --{key}")
                    i += 1
                    continue

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

            parameters[key] = value
            print(f"Override: {key} = {value}")
        i += 1

    main(parameters)
