from __future__ import annotations

import numpy as np
import argparse
import pandas as pd
import os
import glob
import shutil
from tqdm import tqdm
from typing import List, Tuple
import h5py

from ReactBench.utils.frequency_analysis import analyze_frequencies

from pyscf import dft, gto

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
    print(f"Getting geometries and hessians from {args.inp_path}")
    # rxn9_final_hessian_autograd.h5
    autograd_hessians = glob.glob(f"{args.inp_path}/*_{args.which}_hessian_autograd.h5")
    predicted_hessians = glob.glob(f"{args.inp_path}/*_{args.which}_hessian_predict.h5")
    geometries = glob.glob(f"{args.inp_path}/*{args.which}_geometry.xyz")
    # sort by rxn_ind: .../rxn591_ts_final_geometry.xyz
    autograd_hessians.sort(key=lambda x: int(x.split("/")[-1].split("_")[0][3:]))
    predicted_hessians.sort(key=lambda x: int(x.split("/")[-1].split("_")[0][3:]))
    geometries.sort(key=lambda x: int(x.split("/")[-1].split("_")[0][3:]))
    # only keep the rxn_idx that exist in all three lists
    grad_idxs = [int(hess.split("/")[-1].split("_")[0][3:]) for hess in autograd_hessians]
    pred_idxs = [int(hess.split("/")[-1].split("_")[0][3:]) for hess in predicted_hessians]
    geom_idxs = [int(geom.split("/")[-1].split("_")[0][3:]) for geom in geometries]
    print(f"Found {len(grad_idxs)} autograd hessians, {len(pred_idxs)} predicted hessians, and {len(geom_idxs)} geometries")
    rxn_idxs = set(grad_idxs).intersection(set(pred_idxs)).intersection(set(geom_idxs))
    assert len(rxn_idxs) > 0, f"No common rxn_idxs found: {rxn_idxs}"
    autograd_hessians = [hess for hess in autograd_hessians if int(hess.split("/")[-1].split("_")[0][3:]) in rxn_idxs]
    predicted_hessians = [hess for hess in predicted_hessians if int(hess.split("/")[-1].split("_")[0][3:]) in rxn_idxs]
    geometries = [geom for geom in geometries if int(geom.split("/")[-1].split("_")[0][3:]) in rxn_idxs]
    print(f"Found {len(autograd_hessians)} autograd hessians, {len(predicted_hessians)} predicted hessians, and {len(geometries)} geometries")
    assert len(autograd_hessians) == len(predicted_hessians) == len(geometries), (
        f"Number of hessians and geometries do not match: grad={len(autograd_hessians)}, pred={len(predicted_hessians)}, geom={len(geometries)}"
    )
    return geometries, predicted_hessians, autograd_hessians


# pysisyphus
def launch_dft_and_save_hessians(geometries, outdir, args):
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = len(geometries)
    print(f"\nLaunching {max_samples} DFT calculations")
    dft_hessian_paths = []
    for geom_path in tqdm(geometries[:max_samples], desc="DFT Hessian", total=max_samples):
        rxnname = geom_path.split("/")[-1].split("_")[0]  # rxn9
        fname = f"{rxnname}_{args.which}_hessian_dft.h5"
        h5_path = os.path.join(outdir, fname)
        txt_path = h5_path.replace(".h5", ".txt")
        # Check if DFT Hessian already exists and we don't want to redo the calculation
        if os.path.exists(h5_path) and not args.redo:
            dft_hessian_paths.append(h5_path)
            print(f"DFT Hessian already exists for {rxnname}. Appending.")
            continue
        # Check if DFT Hessian previously failed and we want to retry the calculation
        elif os.path.exists(txt_path) and not args.retry_dft:
            dft_hessian_paths.append(None)
            print(f"DFT Hessian previously failed for {rxnname}. Appending None.")
            continue
        # Otherwise, we need to compute the DFT Hessian
        xyz_path = os.path.abspath(geom_path) 
        atoms = read_xyz(xyz_path) # Angstrom
        # convert Angstrom to Bohr
        atoms = [(sym, (x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR)) for sym, (x, y, z) in atoms]
        mol = build_molecule(atoms)
        hessian_dft = compute_hessian(mol, xc="wb97x", debug_hint=rxnname)
        # Handle case where DFT Hessian failed
        if hessian_dft is None:
            dft_hessian_paths.append(None)
            # save a dummy .txt file
            with open(txt_path, "w") as f:
                f.write("DFT Hessian failed")
            print(f"DFT Hessian failed for {rxnname}. Appending None.")
            continue
        else:
            dft_hessian_paths.append(h5_path)
            print(f"Hessian calculation for {rxnname} successful.")
        # To save the Hessian, we need to build a pysisyphus geometry object
        atom_symbols = [atom[0] for atom in atoms]
        coords = np.array([atom[1] for atom in atoms])
        geom = Geometry(
            atoms=atom_symbols, coords=coords.flatten(), 
            coord_type="redund"
        )
        geom.cart_hessian = hessian_dft
        geom.energy = 0.0
        save_h5_hessian(h5_path, geom)
        # Copy geometry file for future reference
        geom_fname = f"{fname.split('_')[0]}_{args.which}_geometry.xyz"
        geom_copy_path = os.path.join(outdir, geom_fname)
        shutil.copy2(xyz_path, geom_copy_path)
    return dft_hessian_paths


# ReactBench/utils/parsers.py
def xyz_parse(input, multiple=False, return_info=False):
    """
    Simple wrapper function for grabbing the coordinates and
    elements from an xyz file.
    Units are Angstrom (same as xyz files).

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

    if not multiple:
        assert N_atoms == len(molecules[0][0]), f"N_atoms={N_atoms}, molecules={molecules}, input={input}"

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

def load_hessian_h5(h5_path):
    """Loads from pysisyphus. Units are pysis default of Hartree/Bohr^2."""
    with h5py.File(h5_path, "r") as handle:
        atoms = [atom.capitalize() for atom in handle.attrs["atoms"]]
        coords3d = handle["coords3d"][:] # Bohr
        energy = handle.attrs["energy"] # Hartree
        cart_hessian = handle["hessian"][:] # Hartree/Bohr^2
    return cart_hessian, atoms, coords3d, energy

def compare_hessians(
    geometries, predicted_hessians, autograd_hessians, dft_hessians, args
):
    # Initialize empty list to collect data
    data_rows = []
    cnt_fail_freq_analysis = 0
    cnt_fail_autograd_predict_hessian_same = 0
    assert len(dft_hessians) > 0, "No dft_hessians found"
    # assert len(geometries) == len(predicted_hessians) == len(autograd_hessians) == len(dft_hessians), (
    #     f"Number of hessians and geometries do not match: {len(geometries)}, {len(predicted_hessians)}, {len(autograd_hessians)}, {len(dft_hessians)}"
    # )

    print("\nComparing number of imaginary frequencies between Hessian methods:")
    for i, (geo, hess_pred, hess_grad, hess_dft) in enumerate(zip(
        geometries, predicted_hessians, autograd_hessians, dft_hessians
    )):
        # Extract reaction index from geometry filename
        rxn_ind = geo.split("/")[-1].split("_")[0]  # rxn9
        if hess_dft is None:
            print(f"DFT Hessian is None for {i}: {rxn_ind}. Skipping...")
            continue
        rxn_ind_pred = hess_pred.split("/")[-1].split("_")[0]
        rxn_ind_grad = hess_grad.split("/")[-1].split("_")[0]
        rxn_ind_dft = hess_dft.split("/")[-1].split("_")[0]
        assert rxn_ind == rxn_ind_pred, f"Reaction index mismatch at {i}: {rxn_ind} != {rxn_ind_pred}"
        assert rxn_ind == rxn_ind_grad, f"Reaction index mismatch at {i}: {rxn_ind} != {rxn_ind_grad}"
        assert rxn_ind == rxn_ind_dft, f"Reaction index mismatch at {i}: {rxn_ind} != {rxn_ind_dft}"
        
        # eigenvalues, wavenumbers, negative eigenvalues, number of negative eigenvalues
        # of Eckart-projection of mass-weighted hessian
        dft_freqs = analyze_frequencies_pysisyphus(hess_dft, xyz_path=geo, debug_hint=f"DFT {rxn_ind}")
        grad_freqs = analyze_frequencies_pysisyphus(hess_grad, xyz_path=geo, debug_hint=f"Autograd {rxn_ind}")
        pred_freqs = analyze_frequencies_pysisyphus(hess_pred, xyz_path=geo, debug_hint=f"Predicted {rxn_ind}")

        # as a sanity check, compare to ReactBench's frequency analysis
        atomsymbols, cart_xyz = xyz_parse(geo) 
        cart_xyz = np.asarray(cart_xyz) * ANG2BOHR
        try:
            dft_freqs_rb = analyze_frequencies(hess_dft, cart_xyz, atomsymbols)
        except Exception as e:
            print(f"Error in ReactBench frequency analysis for {rxn_ind}: {e}")
            cnt_fail_freq_analysis += 1
            continue

        # check that predicted hessian and the autograd hessian are different
        hess_pred_array = load_hessian_h5(hess_pred)[0]
        hess_grad_array = load_hessian_h5(hess_grad)[0]
        if np.allclose(hess_pred_array, hess_grad_array):
            print(f"Error: Predicted and Autograd Hessians are the same for {rxn_ind}. Check how the Hessians were saved. Skipping...")
            cnt_fail_autograd_predict_hessian_same += 1
            continue

        if dft_freqs["neg_num"] != dft_freqs_rb["neg_num"]:
            print(f"DFT frequency analysis mismatch for {rxn_ind}")
            print(
                f"ReactBench: {dft_freqs_rb['neg_num']}, pysisyphus: {dft_freqs['neg_num']}"
            )
            print(
                f"ReactBench: {dft_freqs_rb['eigvals']}, pysisyphus: {dft_freqs['eigvals']}"
            )
            print(
                f"ReactBench: {dft_freqs_rb['neg_eigvals']}, pysisyphus: {dft_freqs['neg_eigvals']}"
            )
            print(
                f"ReactBench: {dft_freqs_rb['natoms']}, pysisyphus: {dft_freqs['natoms']}"
            )
            raise ValueError("DFT frequency analysis mismatch")

        # add nneg, natoms, rxn_ind to the dataframe
        data_rows.append(
            {
                "rxn_ind": rxn_ind,
                "natoms": dft_freqs["natoms"],
                "dft_nneg": dft_freqs["neg_num"],
                "grad_nneg": grad_freqs["neg_num"],
                "pred_nneg": pred_freqs["neg_num"],
                "dft_eigvals": dft_freqs["eigvals"],
                "grad_eigvals": grad_freqs["eigvals"],
                "pred_eigvals": pred_freqs["eigvals"],
                "dft_neg_eigvals": dft_freqs["neg_eigvals"],
                "grad_neg_eigvals": grad_freqs["neg_eigvals"],
                "pred_neg_eigvals": pred_freqs["neg_eigvals"],
            }
        )
        print(f"{rxn_ind}: dft={dft_freqs['neg_num']} grad={grad_freqs['neg_num']} pred={pred_freqs['neg_num']}")

    print("\nDone comparing Hessians with frequency analysis.")
    print(f"Number of failed frequency analyses: {cnt_fail_freq_analysis}")
    print(f"Number of failed cases where predicted and autograd hessians are the same: {cnt_fail_autograd_predict_hessian_same}")

    # Create DataFrame from collected data
    df_results = pd.DataFrame(data_rows)
    
    # Check if we have any valid data
    if df_results.empty:
        raise ValueError("Warning: No valid DFT hessian data found. All DFT calculations may have failed.")
    
    # add new columns to the dataframe
    # is transition state (index-1 saddle): nneg where == 1
    df_results["dft_is_ts"] = df_results["dft_nneg"] == 1
    df_results["grad_is_ts"] = df_results["grad_nneg"] == 1
    df_results["pred_is_ts"] = df_results["pred_nneg"] == 1

    # regard DFT as ground truth reference
    # compute TP, FP, TN, FN for grad and pred

    # For autograd hessians vs DFT
    grad_tp = (df_results["dft_is_ts"] & df_results["grad_is_ts"]).sum()
    grad_fp = (~df_results["dft_is_ts"] & df_results["grad_is_ts"]).sum()
    grad_tn = (~df_results["dft_is_ts"] & ~df_results["grad_is_ts"]).sum()
    grad_fn = (df_results["dft_is_ts"] & ~df_results["grad_is_ts"]).sum()

    # For predicted hessians vs DFT
    pred_tp = (df_results["dft_is_ts"] & df_results["pred_is_ts"]).sum()
    pred_fp = (~df_results["dft_is_ts"] & df_results["pred_is_ts"]).sum()
    pred_tn = (~df_results["dft_is_ts"] & ~df_results["pred_is_ts"]).sum()
    pred_fn = (df_results["dft_is_ts"] & ~df_results["pred_is_ts"]).sum()

    # Calculate derived metrics
    grad_accuracy = (grad_tp + grad_tn) / len(df_results) if len(df_results) > 0 else 0
    grad_precision = grad_tp / (grad_tp + grad_fp) if (grad_tp + grad_fp) > 0 else 0
    grad_recall = grad_tp / (grad_tp + grad_fn) if (grad_tp + grad_fn) > 0 else 0
    grad_f1 = (
        2 * (grad_precision * grad_recall) / (grad_precision + grad_recall)
        if (grad_precision + grad_recall) > 0
        else 0
    )

    pred_accuracy = (pred_tp + pred_tn) / len(df_results) if len(df_results) > 0 else 0
    pred_precision = pred_tp / (pred_tp + pred_fp) if (pred_tp + pred_fp) > 0 else 0
    pred_recall = pred_tp / (pred_tp + pred_fn) if (pred_tp + pred_fn) > 0 else 0
    pred_f1 = (
        2 * (pred_precision * pred_recall) / (pred_precision + pred_recall)
        if (pred_precision + pred_recall) > 0
        else 0
    )

    # Summary statistics
    results = {
        "total_geometries": len(df_results),
        "dft_ts_count": df_results["dft_is_ts"].sum(),
        "pred_ts_count": df_results["pred_is_ts"].sum(),
        "grad_ts_count": df_results["grad_is_ts"].sum(),
        "autograd_stats": {
            "tp": grad_tp,
            "fp": grad_fp,
            "tn": grad_tn,
            "fn": grad_fn,
            "accuracy": grad_accuracy,
            "precision": grad_precision,
            "recall": grad_recall,
            "f1_score": grad_f1,
        },
        "predicted_stats": {
            "tp": pred_tp,
            "fp": pred_fp,
            "tn": pred_tn,
            "fn": pred_fn,
            "accuracy": pred_accuracy,
            "precision": pred_precision,
            "recall": pred_recall,
            "f1_score": pred_f1,
        },
    }

    return df_results, results


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


def analyze_frequencies_pysisyphus(h5_hessian_path, ev_thresh=-1e-6, xyz_path=None, debug_hint=""):
    """
    ev_thresh: threshold for negative eigenvalues

    from
    pysisyphus/pysisyphus/helpers.py
    """
    rxn_ind = h5_hessian_path.split("/")[-1].split("_")[0]  # rxn9
    geom: Geometry = geom_from_hessian(h5_hessian_path) # Bohr and Hartree/Bohr^2
    if xyz_path is not None:
        # compare geometry and Hessian coordinates and make sure they are the same
        atomsymbols, cart_xyz = xyz_parse(xyz_path)
        cart_xyz = np.asarray(cart_xyz) * ANG2BOHR
        if cart_xyz.shape != geom.coords3d.shape:
            print(f"{debug_hint}: XYZ and Hessian coordinates shapes do not match: {cart_xyz.shape} != {geom.coords3d.shape}")
        elif not np.allclose(geom.coords3d, cart_xyz):
            msg = (
                "!"*100,
                f"Error: {debug_hint}: XYZ and Hessian coordinates do not match.",
                f"max diff: {np.abs(geom.coords3d - cart_xyz).max():.1e}",
                f"max diff (Bohr): {np.abs(geom.coords3d - cart_xyz / ANG2BOHR).max():.1e}"
            )
            # print("\n".join(msg))
            raise ValueError("\n".join(msg))
        else:
            # print(f" {debug_hint}: XYZ and Hessian coordinates match {np.abs(geom.coords3d - cart_xyz).max():.1e}. All good.")
            pass
        # check the atomsymbols
        if not np.all(np.array(atomsymbols) == np.array(geom.atoms)):
            msg = f"Error: {debug_hint}: XYZ and Hessian atomsymbols do not match: {atomsymbols} != {geom.atoms}"
            raise ValueError(msg)
        else:
            pass
            # print(f" {debug_hint}: XYZ and Hessian atomsymbols match.") # sanity check
    # geom.set_calculator(calc_getter())
    hessian = geom.cart_hessian
    mw_hessian = geom.mass_weigh_hessian(hessian)
    proj_hessian = geom.eckart_projection(mw_hessian)
    eigvals, _ = np.linalg.eigh(proj_hessian)
    # frequency analysis
    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    # eigval_str = np.array2string(eigvals[:10], precision=4)
    # print("First 10 eigenvalues", eigval_str)
    if neg_num > 0:
        wavenumbers = eigval_to_wavenumber(neg_eigvals)
        # wavenum_str = np.array2string(wavenumbers, precision=2)
        # print("Imaginary frequencies:", wavenum_str, "cm⁻¹")
    else:
        wavenumbers = None
    return {
        "eigvals": eigvals,
        "wavenumbers": wavenumbers,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(eigvals) // 3,
        "rxn_ind": rxn_ind,
    }


def read_xyz(path: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Reads from XYZ file. Units are Angstrom.
    Returns a list of tuples (symbol, (x,y,z)).
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    try:
        natoms = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError(f"Invalid XYZ header in {path}: {exc}") from exc

    atom_lines = lines[1 : ]
    atoms: List[Tuple[str, Tuple[float, float, float]]] = []
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if i == 0:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except:
                # probably extra info like energy
                continue
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: '{ln}'")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception as exc:
            raise ValueError(f"Invalid coordinates in line: '{ln}'") from exc
        atoms.append((sym, (x, y, z)))
    if len(atoms) != natoms:
        raise ValueError(
            f"XYZ atom count mismatch: header says {natoms}, parsed {len(atoms)} in {path}"
        )
    return atoms


def build_molecule(
    atoms: List[Tuple[str, Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
) -> gto.Mole:
    """
    Build a PySCF molecule from a list of atoms [(symbol, (x,y,z)), (symbol, (x,y,z)), ...].
    Expects Bohr coordinates!
    """
    spin = multiplicity - 1  # 2S = multiplicity - 1
    mol = gto.Mole()
    mol.atom = atoms  # list[(symbol, (x,y,z))]
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = "6-31g(d)"
    mol.unit = "Bohr"
    mol.build()
    return mol


def compute_hessian(
    mol: gto.Mole, multiplicity: int = 1, xc: str = "wb97x", debug_hint=""
) -> np.ndarray:
    is_open_shell = multiplicity != 1
    if is_open_shell:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    # Tighten SCF a bit for stability
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.verbose = 0  # Suppress SCF convergence messages
    try:
        mf.kernel()
    except Exception as e:
        print("\n" + ">"*40)
        print(f"Error in SCF: {debug_hint}: \n{e}")
        print("<"*40)
        return None
    if not mf.converged:
        print("\n" + ">"*40)
        print(f"SCF did not converge: {debug_hint}")
        print("<"*40)
        return None

    # Use the generic interface available on the SCF object
    # (N, N, 3, 3) where N is number of atoms
    hessian = mf.Hessian().kernel()

    # Properly reshape from (N, N, 3, 3) to (3N, 3N)
    # Need to transpose axes to get proper ordering: [atom_i, coord_i, atom_j, coord_j]
    N = mol.natm
    hes = hessian.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)

    # hes shape: (3N, 3N)
    # atomic units (Hartree/Bohr^2)
    return hes


def dft_hessian_from_xyz(xyz_path: str) -> None:
    "Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF."
    xyz_path = os.path.abspath(xyz_path)
    atoms = read_xyz(xyz_path) # Angstrom
    # convert Angstrom to Bohr
    atoms = [(sym, (x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR)) for sym, (x, y, z) in atoms]
    mol = build_molecule(atoms)
    hessian_dft = compute_hessian(mol, xc="wb97x")
    return hessian_dft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF."
    )

    parser.add_argument(
        "inp_path",
        type=str,
        default="equiformer_alldatagputwoalphadrop0droppathrate0projdrop0",
        help="Path to the input XYZ geometries.",
    )
    parser.add_argument(
        "--hessian_method",
        type=str,
        default="predict",
        help="Method to use for Hessian. predict or autograd",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="final",
        help="Which hessian to use. final or initial",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the DFT Hessian computation. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--retry_dft",
        type=bool,
        default=False,
        help="Retry the DFT Hessian computation if it previously failed. If true, overwrite existing results.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to process. If None, process all samples.",
    )

    args = parser.parse_args()

    inp_path = args.inp_path
    if os.path.isdir(inp_path):
        pass
    else:
        # assume checkpoint name is passed, search for full path
        # e.g.
        # equiformer_alldatagputwoalphadrop0droppathrate0projdrop0
        # runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/ts_geoms_hessians
        _path = glob.glob(f"runs/{inp_path}*ts1x_{args.hessian_method}/ts_geoms_hessians")
        if len(_path) == 1:
            inp_path = _path[0]
        else:
            raise ValueError(f"No or multiple runs found for {inp_path}: {_path}")
    args.inp_path = os.path.abspath(inp_path)

    runname = args.inp_path.split("/")[-2]
    print(f"Running comparison for {runname}")

    outdir = f"runs_dft/{runname}"
    os.makedirs(outdir, exist_ok=True)

    # torch.manual_seed(42)
    np.random.seed(42)

    paths = get_geometries_and_hessians(args)
    geometries, predicted_hessians, autograd_hessians = paths

    dft_hessians = launch_dft_and_save_hessians(geometries, outdir, args)

    print()
    print(f"Number of geometries: {len(geometries)}")
    print(f"Number of predicted hessians: {len(predicted_hessians)}")
    print(f"Number of autograd hessians: {len(autograd_hessians)}")
    print(f"Number of DFT hessians: {len(dft_hessians)}")

    # check that the reaction indices match
    rxn_ind_geo = [geo.split("/")[-1].split("_")[0] for geo in geometries]
    rxn_ind_pred = [hess.split("/")[-1].split("_")[0] for hess in predicted_hessians]
    rxn_ind_grad = [hess.split("/")[-1].split("_")[0] for hess in autograd_hessians]
    assert np.all(rxn_ind_geo == rxn_ind_pred), f"Reaction index mismatch: {rxn_ind_geo} != {rxn_ind_pred}"
    assert np.all(rxn_ind_geo == rxn_ind_grad), f"Reaction index mismatch: {rxn_ind_geo} != {rxn_ind_grad}"
    # rxn_ind_dft = [hess.split("/")[-1].split("_")[0] for hess in dft_hessians]
    # assert all(rxn_ind_geo == rxn_ind_dft), f"Reaction index mismatch: {rxn_ind_geo} != {rxn_ind_dft}"

    df_results, results = compare_hessians(
        geometries, predicted_hessians, autograd_hessians, dft_hessians, args
    )

    # Print summary statistics
    print("\nComparison Results:")
    print(f"Total geometries analyzed: {results['total_geometries']}")
    print(f"DFT: is transition state: {results['dft_ts_count']}")
    print(f"Predicted: is transition state: {results['pred_ts_count']}")
    print(f"Autograd: is transition state: {results['grad_ts_count']}")
    
    print("\nAutograd Hessian Performance:")
    print(f"  Accuracy: {results['autograd_stats']['accuracy']:.3f}")
    print(f"  Precision: {results['autograd_stats']['precision']:.3f}")
    print(f"  Recall: {results['autograd_stats']['recall']:.3f}")
    print(f"  F1-score: {results['autograd_stats']['f1_score']:.3f}")
    print(
        f"  TP: {results['autograd_stats']['tp']}, FP: {results['autograd_stats']['fp']}"
    )
    print(
        f"  TN: {results['autograd_stats']['tn']}, FN: {results['autograd_stats']['fn']}"
    )
    print("\nPredicted Hessian Performance:")
    print(f"  Accuracy: {results['predicted_stats']['accuracy']:.3f}")
    print(f"  Precision: {results['predicted_stats']['precision']:.3f}")
    print(f"  Recall: {results['predicted_stats']['recall']:.3f}")
    print(f"  F1-score: {results['predicted_stats']['f1_score']:.3f}")
    print(
        f"  TP: {results['predicted_stats']['tp']}, FP: {results['predicted_stats']['fp']}"
    )
    print(
        f"  TN: {results['predicted_stats']['tn']}, FN: {results['predicted_stats']['fn']}"
    )

    # Save results to CSV
    results_path = f"{outdir}/results_{args.which}.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    plot_comparison(args)

    print("\nDone!")

    # dft_hessian_from_xyz("data/hessiantest/ts_test.xyz")
