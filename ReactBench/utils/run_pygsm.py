"""
Wrapper to the pyGSM using ASE

"""

import os, sys, argparse
import time, textwrap
import numpy as np

# Add the ReactBench package to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: utils -> ReactBench -> ReactBench (package root)
package_root = os.path.dirname(os.path.dirname(current_dir))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from ase import Atoms
from ase.io import read, write

from pyGSM.coordinate_systems.delocalized_coordinates import (
    DelocalizedInternalCoordinates,
)
from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.growing_string_methods import DE_GSM
from pyGSM.level_of_theories.ase import ASELoT
from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
from pyGSM.optimizers.lbfgs import lbfgs
from pyGSM.potential_energy_surfaces import PES
from pyGSM.utilities import nifty, manage_xyz
from pyGSM.utilities.elements import ElementData
from pyGSM.molecule import Molecule
from pyGSM.utilities.cli_utils import plot


from ReactBench.Calculators import get_calculator, AVAILABLE_CALCULATORS
from ReactBench.utils.parsers import xyz_parse


def str2dict(v):
    lst = v.split(",")
    return {"spin": int(lst[0]), "charge": int(lst[1])}


def parse_arguments(verbose=True):
    parser = argparse.ArgumentParser(
        description="Reaction path transition state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
                Example of use:
                --------------------------------
                python run_pygsm.py -xyzfile yourfile.xyz -package -calc xTB -ID 1
                """
        ),
    )
    parser.add_argument(
        "-xyzfile", help="XYZ file containing reactant and product.", required=True
    )
    parser.add_argument(
        "-calc",
        default="leftnet",
        type=str,
        help=f'select a calculator from {", ".join(AVAILABLE_CALCULATORS)}',
        required=False,
    )
    parser.add_argument(
        "-ID",
        default=0,
        type=int,
        help="string identification number (default: %(default)s)",
        required=False,
    )
    parser.add_argument(
        "-num_nodes",
        type=int,
        default=9,
        help="number of nodes for string (defaults: 9 DE-GSM, 20 SE-GSM)",
        required=False,
    )
    parser.add_argument(
        "-optimizer",
        type=str,
        default="eigenvector_follow",
        help="The optimizer object. (default: %(default)s Recommend LBFGS for large molecules >1000 atoms)",
        required=False,
    )
    parser.add_argument(
        "-opt_print_level",
        type=int,
        default=1,
        help="Printout for optimization. 2 prints everything in opt.",
        required=False,
    )
    parser.add_argument(
        "-gsm_print_level",
        type=int,
        default=1,
        help="Printout for gsm. 1 prints ?",
        required=False,
    )
    parser.add_argument(
        "-xyz_output_format",
        type=str,
        default="molden",
        help="Format of the produced XYZ files",
        required=False,
    )
    parser.add_argument(
        "-linesearch",
        type=str,
        default="NoLineSearch",
        help="default: %(default)s",
        choices=["NoLineSearch", "backtrack"],
    )
    parser.add_argument(
        "-coordinate_type",
        type=str,
        default="TRIC",
        help="Coordinate system (default %(default)s)",
        choices=["TRIC", "DLC", "HDLC"],
    )
    parser.add_argument(
        "-ADD_NODE_TOL",
        type=float,
        default=0.01,
        help="Convergence tolerance for adding new node (default: %(default)s)",
        required=False,
    )
    parser.add_argument(
        "-CONV_TOL",
        type=float,
        default=0.0005,
        help="Convergence tolerance for optimizing nodes (default: %(default)s)",
        required=False,
    )
    parser.add_argument(
        "-growth_direction",
        type=int,
        default=0,
        help="Direction adding new nodes (default: %(default)s)",
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "-reactant_geom_fixed",
        action="store_true",
        help="Fix reactant geometry i.e. do not pre-optimize",
    )
    parser.add_argument(
        "-product_geom_fixed",
        action="store_true",
        help="Fix product geometry i.e. do not pre-optimize",
    )
    parser.add_argument(
        "-nproc",
        type=int,
        default=1,
        help="Processors for calculation. Python will detect OMP_NUM_THREADS, only use this if you want to force the number of processors",
    )
    parser.add_argument(
        "-max_gsm_iters",
        type=int,
        default=100,
        help="The maximum number of GSM cycles (default: %(default)s)",
    )
    parser.add_argument(
        "-max_opt_steps",
        type=int,
        default=3,
        help="The maximum number of node optimizations per GSM cycle (defaults: 3 DE-GSM, 20 SE-GSM)",
    )
    parser.add_argument("-restart_file", help="restart file", type=str)
    parser.add_argument(
        "-conv_Ediff",
        default=100.0,
        type=float,
        help="Energy difference convergence of optimization.",
    )
    parser.add_argument(
        "-conv_dE", default=1.0, type=float, help="State difference energy convergence"
    )
    parser.add_argument(
        "-conv_gmax", default=100.0, type=float, help="Max grad rms threshold"
    )
    parser.add_argument("-DMAX", default=0.1, type=float, help="")
    parser.add_argument(
        "-reparametrize",
        action="store_true",
        help="Reparametrize restart string equally along path",
    )
    parser.add_argument("-interp_method", default="DLC", type=str, help="")
    parser.add_argument(
        "-start_climb_immediately",
        action="store_true",
        help="Start climbing immediately when restarting.",
    )
    parser.add_argument("-info", help="info with spin and charge", type=str2dict)
    parser.add_argument(
        "-device",
        default="cpu",
        type=str,
        help="device for MLFF calculator (cpu/cuda)",
        required=False,
    )

    # ASE calculator's options
    args = parser.parse_args()

    if verbose:
        print_msg()

    # check input using AVAILABLE_CALCULATORS from __init__.py
    valid_calcs = AVAILABLE_CALCULATORS
    if args.calc.lower() not in valid_calcs:
        sys.exit(f"Only supports the following calculators: {', '.join(valid_calcs)}")

    inpfileq = {
        # LOT
        "xyzfile": args.xyzfile,
        "info": args.info,
        "calc": args.calc,
        "coordinate_type": args.coordinate_type,
        "nproc": args.nproc,
        "device": getattr(args, "device", "cpu"),
        # optimizer
        "optimizer": args.optimizer,
        "opt_print_level": args.opt_print_level,
        "linesearch": args.linesearch,
        "DMAX": args.DMAX,
        # output
        "xyz_output_format": args.xyz_output_format,
        # GSM
        "reactant_geom_fixed": args.reactant_geom_fixed,
        "product_geom_fixed": args.product_geom_fixed,
        "num_nodes": args.num_nodes,
        "ADD_NODE_TOL": args.ADD_NODE_TOL,
        "CONV_TOL": args.CONV_TOL,
        "conv_Ediff": args.conv_Ediff,
        "conv_dE": args.conv_dE,
        "conv_gmax": args.conv_gmax,
        "growth_direction": args.growth_direction,
        "ID": args.ID,
        "gsm_print_level": args.gsm_print_level,
        "max_gsm_iters": args.max_gsm_iters,
        "max_opt_steps": args.max_opt_steps,
        # newly added args that did not live here yet
        "restart_file": args.restart_file,
        "interp_method": args.interp_method,
        "reparametrize": args.reparametrize,
        "start_climb_immediately": args.start_climb_immediately,
    }

    return inpfileq


def wrapper_de_gsm(
    atoms_reactant: Atoms,
    atoms_product: Atoms,
    calc,
    optimizer_method="eigenvector_follow",
    coordinate_type="TRIC",
    line_search="NoLineSearch",  # OR: 'backtrack'
    only_climb=False,
    step_size_cap=0.1,  # DMAX in the other wrapper
    num_nodes=9,  # 20 for SE-GSM
    add_node_tol=0.1,  # convergence for adding new nodes
    conv_tol=0.001,  # Convergence tolerance for optimizing nodes
    conv_Ediff=100.0,  # Energy difference convergence of optimization.
    conv_gmax=100.0,  # Max grad rms threshold
    ID=0,
    nproc=1,
    max_gsm_iterations=100,
    max_opt_steps=5,  # 20 for SE-GSM
    reparametrize=True,
    start_climb_immediately=False,
    fixed_reactant=False,
    fixed_product=False,
    restart_file=False,
    info=None,
    device="cpu",
):
    # PES
    # pes_type = "PES"
    # 'PES_type': args.pes_type,
    # 'adiabatic_index': args.adiabatic_index,
    # 'multiplicity': args.multiplicity,
    # 'FORCE_FILE': args.FORCE_FILE,
    # 'RESTRAINT_FILE': args.RESTRAINT_FILE,
    # 'FORCE': None,
    # 'RESTRAINTS': None,

    # 'hybrid_coord_idx_file': args.hybrid_coord_idx_file,
    # 'frozen_coord_idx_file': args.frozen_coord_idx_file,
    # 'prim_idx_file': args.prim_idx_file,

    # GSM
    # gsm_type = "DE_GSM"  # SE_GSM, SE_Cross
    # 'isomers_file': args.isomers,   # driving coordinates, this is a file for SE-GSM
    # 'BDIST_RATIO': args.BDIST_RATIO,
    # 'DQMAG_MAX': args.DQMAG_MAX,
    # 'growth_direction': args.growth_direction,
    # 'gsm_print_level': args.gsm_print_level,
    # 'use_multiprocessing': args.use_multiprocessing,
    nifty.printcool("Parsed GSM")

    if calc.lower() in AVAILABLE_CALCULATORS:
        calculator = get_calculator(calc.lower(), device=device)

    else:
        raise ValueError(
            f"Unknown calculator: {calc}. Only supports: {', '.join(AVAILABLE_CALCULATORS)}"
        )

    # LOT
    if calc.lower() in ["chg", "mattersim"]:
        cell = [100, 100, 100]
    else:
        cell = None
    lot = ASELoT.from_options(
        calculator,
        nproc=nproc,
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        cell=cell,
        ID=ID,
    )

    # PES
    pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)

    # load the initial string
    if restart_file:
        geoms = manage_xyz.read_molden_geoms(restart_file)
    else:
        geoms = [atoms_reactant, atoms_product]

    # Build the topology
    nifty.printcool("Building the topologies")
    element_table = ElementData()
    elements = [
        element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()
    ]

    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(), atoms=elements
    )

    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(), atoms=elements
    )

    # Union of bonds
    # debated if needed here or not
    for bond in topology_product.edges():
        if (
            bond in topology_reactant.edges()
            or (bond[1], bond[0]) in topology_reactant.edges()
        ):
            continue
        print(" Adding bond {} to reactant topology".format(bond))
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])

    # primitive internal coordinates
    nifty.printcool("Building Primitive Internal Coordinates")
    connect = False
    addtr = False
    addcart = False
    if coordinate_type == "DLC":
        connect = True
    elif coordinate_type == "TRIC":
        addtr = True
    elif coordinate_type == "HDLC":
        addcart = True

    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
    )

    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
    )

    # add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # Delocalised internal coordinates
    nifty.printcool("Building Delocalized Internal Coordinates")
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
        primitives=prim_reactant,
    )

    # Molecules
    nifty.printcool("Building the reactant object with {}".format(coordinate_type))
    from_hessian = optimizer_method == "eigenvector_follow"

    molecule_reactant = Molecule.from_options(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=from_hessian,
    )

    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False,
    )

    # optimizer
    nifty.printcool("Building the Optimizer object")
    opt_options = dict(
        print_level=1,
        Linesearch=line_search,
        update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
        conv_Ediff=conv_Ediff,
        conv_gmax=conv_gmax,
        DMAX=step_size_cap,
        opt_climb=only_climb,
    )
    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(**opt_options)
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(**opt_options)
    else:
        raise NotImplementedError

    # GSM
    nifty.printcool("Building the GSM object")
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_Ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,  # normal/react/prod: 0/1/2
        optimizer=optimizer_object,
        ID=ID,
        print_level=1,
        interp_method="DLC",
    )

    # optimize reactant and product if needed
    if not fixed_reactant:
        nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), "scratch", f"{ID:03}", "0")
        optimizer_object.optimize(
            molecule=molecule_reactant,
            refE=molecule_reactant.energy,
            opt_steps=100,
            path=path,
        )
    if not fixed_product:
        nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), "scratch", f"{ID:03}", str(num_nodes - 1))
        optimizer_object.optimize(
            molecule=molecule_product,
            refE=molecule_product.energy,
            opt_steps=100,
            path=path,
        )

    # set 'rtype' as in main one (???)
    if only_climb:
        rtype = 1
    # elif no_climb:
    #     rtype = 0
    else:
        rtype = 2

    # do GSM
    if restart_file:
        nifty.printcool("Restarting GSM Calculation")
        gsm.setup_from_geometries(
            geoms,
            reparametrize=reparametrize,
            start_climb_immediately=start_climb_immediately,
        )
    else:
        nifty.printcool("Main GSM Calculation")

    gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)

    # write the results into an extended xyz file
    string_ase, ts_ase = gsm_to_ase_atoms(gsm)
    write(f"opt_converged_{gsm.ID:03d}_ase.xyz", string_ase)
    write(f"TSnode_{gsm.ID}.xyz", string_ase)

    # post processing taken from the main wrapper, plots as well
    post_processing(gsm, have_TS=True)

    # cleanup
    cleanup_scratch(gsm.ID)


def gsm_to_ase_atoms(gsm: DE_GSM):
    # string
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        at.info["energy"] = energy
        frames.append(at)

    # TS
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(
        symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom]
    )

    return frames, ts_atoms


def post_processing(gsm, analyze_ICs=False, have_TS=True):
    plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.ID)

    ICs = []
    ICs.append(gsm.nodes[0].primitive_internal_coordinates)

    # TS energy
    if have_TS:
        minnodeR = np.argmin(gsm.energies[: gsm.TSnode])
        TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        print(" TS energy: %5.4f" % TSenergy)
        print(" absolute energy TS node %5.4f" % gsm.nodes[gsm.TSnode].energy)
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode :])
        print(
            " min reactant node: %i min product node %i TS node is %i"
            % (minnodeR, minnodeP, gsm.TSnode)
        )

        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open("IC_data_{:03d}.txt".format(gsm.ID), "w") as f:
            f.write(
                "Internals \t minnodeR: {} \t TSnode: {} \t minnodeP: {}\n".format(
                    minnodeR, gsm.TSnode, minnodeP
                )
            )
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    else:
        minnodeR = 0
        minnodeP = gsm.nR
        print(" absolute energy end node %5.4f" % gsm.nodes[gsm.nR].energy)
        print(" difference energy end node %5.4f" % gsm.nodes[gsm.nR].difference_energy)
        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open("IC_data_{}.txt".format(gsm.ID), "w") as f:
            f.write(
                "Internals \t Beginning: {} \t End: {}".format(
                    minnodeR, gsm.TSnode, minnodeP
                )
            )
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\n".format(*x))

    # Delta E
    deltaE = gsm.energies[minnodeP] - gsm.energies[minnodeR]
    print(" Delta E is %5.4f" % deltaE)


def cleanup_scratch(ID):
    cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)


def print_msg():
    msg = """
    __        __   _                            _        
    \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
     \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
      \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
       \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
                                    ____ ____  __  __ 
                       _ __  _   _ / ___/ ___||  \/  |
                      | '_ \| | | | |  _\___ \| |\/| |
                      | |_) | |_| | |_| |___) | |  | |
                      | .__/ \__, |\____|____/|_|  |_|
                      |_|    |___/                    
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |# 
#| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
#| intersection mediated photochemistry with growing string methods",     |#
#| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
#| http://dx.doi.org/10.1039/c8cp04703k                                   |#
#|                                                                        |# 
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#


    """
    print(msg)


def main():

    # argument parsing and header
    inpfileq = parse_arguments(verbose=True)

    # load input rxn
    # input_rxn = read(inpfileq["xyzfile"], ":")
    mols = xyz_parse(inpfileq["xyzfile"], multiple=True)
    reactant = Atoms(symbols=mols[0][0], positions=mols[0][1])
    product = Atoms(symbols=mols[1][0], positions=mols[1][1])

    # Update atoms info if provided
    if inpfileq["info"] is not None:
        reactant.info.update(inpfileq["info"])
        product.info.update(inpfileq["info"])

    # set set_initial_charges and set_initial_magnetic_moments
    atoms_num = len(reactant.get_atomic_numbers())
    initial_charges = np.zeros(atoms_num, dtype=float)
    initial_charges[0] = reactant.info["charge"]
    initial_magnetic_moments = np.zeros(atoms_num, dtype=float)
    initial_magnetic_moments[0] = reactant.info["spin"] - 1

    reactant.set_initial_charges(initial_charges)
    reactant.set_initial_magnetic_moments(initial_magnetic_moments)
    product.set_initial_charges(initial_charges)
    product.set_initial_magnetic_moments(initial_magnetic_moments)

    wrapper_de_gsm(
        reactant,
        product,
        inpfileq["calc"],
        optimizer_method=inpfileq["optimizer"],
        coordinate_type=inpfileq["coordinate_type"],
        line_search=inpfileq["linesearch"],
        step_size_cap=inpfileq["DMAX"],  # DMAX in the other wrapper
        num_nodes=inpfileq["num_nodes"],  # 20 for SE-GSM
        add_node_tol=inpfileq["ADD_NODE_TOL"],  # convergence for adding new nodes
        conv_tol=inpfileq["CONV_TOL"],  # Convergence tolerance for optimizing nodes
        conv_Ediff=inpfileq[
            "conv_Ediff"
        ],  # Energy difference convergence of optimization.
        conv_gmax=inpfileq["conv_gmax"],  # Max grad rms threshold
        ID=inpfileq["ID"],
        nproc=inpfileq["nproc"],
        max_gsm_iterations=inpfileq["max_gsm_iters"],
        max_opt_steps=inpfileq["max_opt_steps"],  # 20 for SE-GSM
        reparametrize=inpfileq["reparametrize"],
        start_climb_immediately=inpfileq["start_climb_immediately"],
        fixed_reactant=inpfileq["reactant_geom_fixed"],
        fixed_product=inpfileq["product_geom_fixed"],
        restart_file=inpfileq["restart_file"],
        info=inpfileq["info"],
        device=inpfileq.get("device", "cpu"),
    )

    return


if __name__ == "__main__":
    main()
