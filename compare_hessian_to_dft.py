import numpy as np
import argparse
import h5py
from tqdm import tqdm
import wandb
import pandas as pd
import plotly.graph_objects as go
import time
import sys
import json
from pathlib import Path
import os

from pysisyphus.config import p_DEFAULT, T_DEFAULT, LIB_DIR
from pysisyphus.constants import ANG2BOHR, AU2KJPERMOL
from pysisyphus.Geometry import Geometry
from pysisyphus.helpers_pure import (
    eigval_to_wavenumber,
    report_isotopes,
    highlight_text,
    rms,
)
from pysisyphus.io import (
    geom_from_cjson,
    geom_from_crd,
    geom_from_cube,
    geom_from_fchk,
    geom_from_hessian,
    geom_from_mol2,
    geom_from_pdb,
    geom_from_qcschema,
    save_hessian as save_h5_hessian,
    geom_from_zmat_fn,
    geoms_from_inline_xyz,
    geom_from_pubchem_name,
)


def get_geometries_and_hessians(args):
    pass


def run_dft(geometries, args):
    pass


# ReactBench/utils/parsers.py
def xyz_parse(input, multiple=False, return_info=False):
    """
    Simple wrapper function for grabbing the coordinates and
    elements from an xyz file

    Inputs      input: string holding the filename of the xyz
    Returns     (List of) Elements: list of element types (list of strings)
                (List of) Geometry: Nx3 array holding the cartesian coordinates of the
                                    geometry (atoms are indexed to the elements in Elements)
    """

    # Initialize a list to hold all molecule data
    molecules = []
    infos = []

    # Open and read the input file
    with open(input, "r") as f:
        lines = f.readlines()

    # Initialize molecule information
    N_atoms = 0
    Elements = []
    Geometry = np.array([])  # Init as empty array
    count = 0
    comment_lc = 1

    # Iterate over all lines in the file
    for lc, line in enumerate(lines):
        fields = line.split()

        # New molecule begins
        if lc == 0 or (len(fields) == 1 and fields[0].isdigit()):
            # If a molecule has been read, append it to the list
            if Elements and not Geometry.size == 0:
                molecules.append((Elements, Geometry))

            # Reset molecule information for new molecule
            if len(fields) == 1 and fields[0].isdigit():
                N_atoms = int(fields[0])
                Elements = ["X"] * N_atoms
                Geometry = np.zeros([N_atoms, 3])
                count = 0
                comment_lc = lc + 1
                continue

        # parse comments
        if lc == comment_lc:
            infos.append(line.strip())
            continue

        # If it's not a new molecule or an empty line, continue parsing atom data
        if len(fields) == 4:
            try:
                x = float(fields[1])
                y = float(fields[2])
                z = float(fields[3])
                # If we reach this point, the conversion worked and we can save the data
                Elements[count] = fields[0]
                Geometry[count, :] = np.array([x, y, z])
                count += 1
            except ValueError:
                # If the conversion failed, the line did not contain valid atom data, so we skip it
                continue

    # Append the last molecule
    if Elements and not Geometry.size == 0:
        molecules.append((Elements, Geometry))

    if return_info:
        if not multiple:
            return molecules[0], infos[0]
        else:
            return molecules, infos
    else:
        if not multiple:
            return molecules[0]
        else:
            return molecules


def compare_hessians(
    geometries, predicted_hessians, autograd_hessians, dft_hessians, args
):
    for geo, hess_pred, hess_grad, hess_dft in zip(
        geometries, predicted_hessians, autograd_hessians, dft_hessians
    ):
        # eigenvalues, wavenumbers, negative eigenvalues, number of negative eigenvalues
        # of Eckart-projection of mass-weighted hessian
        eigvals_dft, wn_dft, negeigvals_dft, nneg_dft = frequency_analysis(hess_dft)
        eigvals_grad, wn_grad, negeigvals_grad, nneg_grad = frequency_analysis(
            hess_grad
        )
        eigvals_pred, wn_pred, negeigvals_pred, nneg_pred = frequency_analysis(
            hess_pred
        )

        # check how many are transition state (index-1 saddle)
        ists_dft = np.where(np.asarray(nneg_dft) == 1, 1, 0)
        ists_grad = np.where(np.asarray(nneg_grad) == 1, 1, 0)
        ists_pred = np.where(np.asarray(nneg_pred) == 1, 1, 0)


def plot_comparison(args):
    pass


# def geom_from_hessian(h5_fn, **geom_kwargs):
#     with h5py.File(h5_fn, "r") as handle:
#         atoms = [atom.capitalize() for atom in handle.attrs["atoms"]]
#         coords3d = handle["coords3d"][:]
#         energy = handle.attrs["energy"]
#         cart_hessian = handle["hessian"][:]

#     geom = Geometry(atoms=atoms, coords=coords3d, **geom_kwargs)
#     geom.cart_hessian = cart_hessian
#     geom.energy = energy
#     return geom


def frequency_analysis(h5_hessian_path, ev_thresh=-1e-6):
    """
    ev_thresh: threshold for negative eigenvalues

    from
    pysisyphus/pysisyphus/helpers.py
    """
    geom: Geometry = geom_from_hessian(h5_hessian_path)
    # geom.set_calculator(calc_getter())
    print("... started Hessian calculation")
    hessian = geom.cart_hessian
    print("... mass-weighing cartesian hessian")
    mw_hessian = geom.mass_weigh_hessian(hessian)
    print("... doing Eckart-projection")
    proj_hessian = geom.eckart_projection(mw_hessian)
    eigvals, _ = np.linalg.eigh(proj_hessian)

    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    eigval_str = np.array2string(eigvals[:10], precision=4)
    print()
    print("First 10 eigenvalues", eigval_str)
    if neg_num > 0:
        wavenumbers = eigval_to_wavenumber(neg_eigvals)
        wavenum_str = np.array2string(wavenumbers, precision=2)
        print("Imaginary frequencies:", wavenum_str, "cm⁻¹")
        print("neg_num:", neg_num)
    return eigvals, wavenumbers, neg_eigvals, neg_num


if __name__ == "__main__":
    """
    python scripts/speed_comparison.py speed --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
    python scripts/speed_comparison.py speed --dataset ts1x-val.lmdb --max_samples_per_n 100
    python scripts/speed_comparison.py speed --dataset ts1x_hess_train_big.lmdb --max_samples_per_n 1000
    """
    parser = argparse.ArgumentParser(
        description="HORM model evaluation and speed comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for evaluation
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Compare frequency analysis of predicted and DFT Hessians at found transition states",
    )

    eval_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name",
    )
    eval_parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    eval_parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )

    args = parser.parse_args()

    # torch.manual_seed(42)
    np.random.seed(42)

    paths = get_geometries_and_hessians(args)
    geometries, predicted_hessians, autograd_hessians = paths

    dft_hessians = run_dft(geometries, args)

    compare_hessians(
        geometries, predicted_hessians, autograd_hessians, dft_hessians, args
    )
    plot_comparison(args)

    print("\nDone!")
