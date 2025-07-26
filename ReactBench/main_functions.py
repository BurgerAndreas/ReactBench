import os
import numpy as np
import logging
from ReactBench.utils.parsers import xyz_parse, xyz_write, mol_write
from ReactBench.utils.find_lewis import find_lewis
from ReactBench.utils.taffi_functions import table_generator

from openbabel import pybel


def return_smi(E, G, bond_mat=None, namespace="obabel"):
    """Function to Return smiles string using openbabel (pybel)"""
    if bond_mat is None:
        xyz_write(f"{namespace}_input.xyz", E, G)
        # Read the XYZ file using Open Babel
        molecule = next(pybel.readfile("xyz", f"{namespace}_input.xyz"))
        # Generate the canonical SMILES string directly
        smile = molecule.write(format="can").strip().split()[0]
        # Clean up the temporary file
        os.remove(f"{namespace}_input.xyz")
        return smile

    else:
        mol_write(f"{namespace}_input.mol", E, G, bond_mat)
        # Read the mol file using Open Babel
        molecule = next(pybel.readfile("mol", f"{namespace}_input.mol"))
        # Generate the canonical SMILES string directly
        smile = molecule.write(format="can").strip().split()[0]
        # Clean up the temporary file
        os.remove(f"{namespace}_input.mol")

    return smile


def logger_process(queue, logging_path):
    """A child process for logging all information from other processes"""
    logger = logging.getLogger("YARPrun")
    logger.addHandler(logging.FileHandler(logging_path))
    logger.setLevel(logging.INFO)
    while True:
        try:
            message = queue.get()
            if message is None:  # Sentinel value to terminate the logger process
                break
            logger.handle(message)
        except EOFError:
            logger.error("Logger process encountered an EOFError")
            break


def analyze_outputs(
    working_folder,
    irc_job_list,
    logger,
    charge=0,
    dg_thresh=None,
    uncertainty=5,
    select="tight",
    use_BE=True,
):
    """
    Analyze the first stage YARP calculation, find intended&unintended reactions, etc
    Input: dg_thresh -- threshold for activation energy, above which no further DFT analysis will be needed (default: None)
           uncertainty -- trust region of low-level calculations
           working_folder -- output files, including IRC_record.txt, selected_tss.txt, will be stored in this folder
           irc_job_list -- a list of irc jobs
    """
    # initialize output dictionary
    reactions = dict()

    # create record.txt to write the IRC result
    with open(f"{working_folder}/IRC-record.txt", "w") as g:
        g.write(
            f"{'reaction':40s} {'R':<60s} {'P':<60s} {'type':<15s} {'barrier':<10s}\n"
        )

    # loop over IRC output files
    for irc_job in irc_job_list:
        # check job status
        job_success = False
        rxn_ind = irc_job.jobname
        try:
            E, G1, G2, TSG, barrier1, barrier2, TSE = irc_job.analyze_IRC()
            barriers = [barrier1, barrier2]
            job_success = True
        except:
            pass

        if job_success is False:
            logger.info(f"IRC job {irc_job.jobname} fails, skip this reaction")
            print(f"IRC job {irc_job.jobname} fails, skip this reaction")
            continue

        # obtain the original input reaction
        input_xyz = f"{working_folder}/init_rxns/{rxn_ind}.xyz"
        [[_, RG], [_, PG]] = xyz_parse(input_xyz, multiple=True)

        # compute adjacency matrix
        adjmat_1, adjmat_2, R_adjmat, P_adjmat = (
            table_generator(E, G1),
            table_generator(E, G2),
            table_generator(E, RG),
            table_generator(E, PG),
        )

        # if no bond change happens, skip this reaction
        adj_diff = np.abs(adjmat_1 - adjmat_2)
        if adj_diff.sum() == 0:
            print(f"{rxn_ind} is a conformation change, skip this TS...")
            continue

        # compute bond matrix if is required, then obtain smiles
        if use_BE:
            bond_mats_1, _ = find_lewis(E, adjmat_1, charge)
            bond_mats_2, _ = find_lewis(E, adjmat_2, charge)
            smiles = [
                return_smi(E, G1, bond_mats_1[0], namespace=f"{rxn_ind}-1"),
                return_smi(E, G2, bond_mats_2[0], namespace=f"{rxn_ind}-2"),
            ]
        else:
            smiles = [
                return_smi(E, G1, namespace=f"{rxn_ind}-1"),
                return_smi(E, G2, namespace=f"{rxn_ind}-2"),
            ]

        # compare adj_mats
        adj_diff_1r = np.abs(adjmat_1 - R_adjmat)
        adj_diff_1p = np.abs(adjmat_1 - P_adjmat)
        adj_diff_2r = np.abs(adjmat_2 - R_adjmat)
        adj_diff_2p = np.abs(adjmat_2 - P_adjmat)

        # ignore the connectivity differences on metal
        ignore_index = [
            ind
            for ind, ele in enumerate(E)
            if ele.lower() in ["zn", "mg", "li", "si", "ag", "cu", "b"]
        ]
        for ind in ignore_index:
            adj_diff_1r[ind, :] = adj_diff_1r[:, ind] = 0
            adj_diff_2r[ind, :] = adj_diff_2r[:, ind] = 0
            adj_diff_1p[ind, :] = adj_diff_1p[:, ind] = 0
            adj_diff_2p[ind, :] = adj_diff_2p[:, ind] = 0

        # match two IRC end nodes
        rtype = "Unintended"
        node_map = {"R": None, "P": None}

        if adj_diff_1r.sum() == 0:
            if adj_diff_2p.sum() == 0:
                rtype = "Intended"
                node_map = {"R": 1, "P": 2}
            else:
                rtype = "P_Unintended"
                node_map = {"R": 1, "P": 2}

        elif adj_diff_1p.sum() == 0:
            if adj_diff_2r.sum() == 0:
                rtype = "Intended"
                node_map = {"R": 2, "P": 1}
            else:
                rtype = "R_Unintended"
                node_map = {"R": 2, "P": 1}
        elif adj_diff_2r.sum() == 0:
            rtype = "P_Unintended"
            node_map = {"R": 2, "P": 1}
        elif adj_diff_2p.sum() == 0:
            rtype = "R_Unintended"
            node_map = {"R": 1, "P": 2}

        # analyze outputs
        if rtype == "Intended":
            rsmiles = smiles[node_map["R"] - 1]
            psmiles = smiles[node_map["P"] - 1]
            barrier = barriers[node_map["R"] - 1]
        elif rtype == "P_Unintended":
            rsmiles = smiles[node_map["R"] - 1]
            psmiles = smiles[node_map["P"] - 1]
            barrier = barriers[node_map["R"] - 1]
        elif rtype == "R_Unintended":
            rsmiles = smiles[node_map["R"] - 1]
            psmiles = smiles[node_map["P"] - 1]
            barrier = barriers[node_map["P"] - 1]
        else:
            rsmiles = smiles[0]
            psmiles = smiles[1]
            node_map = {"R": 1, "P": 2}
            barrier = barriers[node_map["R"] - 1]

        ### select based on input settings
        rxn_smi_r = f"{rsmiles}>>{psmiles}"
        reactions[rxn_ind] = {
            "rxn": rxn_smi_r,
            "barrier": barrier,
            "TS_SPE": TSE,
            "rtype": rtype,
            "reactant": rsmiles,
            "product": psmiles,
            "select": 1,
        }

        # select based on the using scenario
        if select == "network":
            if rtype not in ["Intended", "P_Unintended"]:
                reactions[rxn_ind]["select"] = 0
                continue
        elif select == "tight":
            if rtype != "Intended":
                reactions[rxn_ind]["select"] = 0
                continue
        elif select == "loose":
            if rtype not in ["Intended", "R_Unintended", "P_Unintended", "Unintended"]:
                reactions[rxn_ind]["select"] = 0
                continue

        # in case IRC calculation is wrong, print out warning information and ignore this rxn
        if reactions[rxn_ind]["select"] and barrier < 0:
            print(
                f"Reaction {irc_job.jobname} has a barrier less than 0, which indicates the end node optimization of IRC has some trouble, please manually check this reaction"
            )
            logger.info(
                f"Reaction {irc_job.jobname} has a barrier less than 0, which indicates the end node optimization of IRC has some trouble, please manually check this reaction"
            )
            reactions[rxn_ind]["select"] = 0
            continue

        # if there is a dg_thresh, the barrier needs to be smaller than the threshold plus the soft margin to be selected
        if dg_thresh is not None and barrier > dg_thresh + uncertainty:
            print(
                f"Reaction {irc_job.jobname} has a barrier of {barrier} kcal/mol which is higher than the barrier threshold"
            )
            logger.info(
                f"Reaction {irc_job.jobname} has a barrier of {barrier} kcal/mol which is higher than the barrier threshold"
            )
            reactions[rxn_ind]["select"] = 0
            continue

    # write down reaction informations into IRC output file
    for rxn_ind in sorted(reactions.keys()):
        rxn = reactions[rxn_ind]
        with open(f"{working_folder}/IRC-record.txt", "a") as g:
            g.write(
                f"{rxn_ind:40s} {rxn['reactant']:60s} {rxn['product']:60s} {rxn['rtype']:15s} {str(rxn['barrier']):10s}\n"
            )
