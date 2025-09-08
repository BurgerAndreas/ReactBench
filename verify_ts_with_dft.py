from __future__ import annotations

import numpy as np
import argparse
import pandas as pd
import os
import glob
from tqdm import tqdm
from typing import List, Tuple
import h5py
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import json

from ReactBench.utils.frequency_analysis import analyze_frequencies, ANG2BOHR

import pyscf
from pyscf import gto

try:
    from gpu4pyscf.dft import rks

    pyscf_on_gpu = True
    print("!Using GPU4PySCF!")
except ImportError as e:
    print(e)
    print("Using CPU-based PySCF")
    from pyscf import dft
    from pyscf.dft import rks

    pyscf_on_gpu = False

# Set default Plotly theme
pio.templates.default = "plotly_white"


# ANG2BOHR imported from ReactBench.utils.frequency_analysis to avoid external dependency
# from pysisyphus.config import p_DEFAULT, T_DEFAULT, LIB_DIR
# from pysisyphus.Geometry import Geometry
# from pysisyphus.helpers_pure import (
#     eigval_to_wavenumber,
#     report_isotopes,
#     highlight_text,
#     rms,
# )
# from pysisyphus.io import (
#     geom_from_cjson,
#     geom_from_crd,
#     geom_from_cube,
#     geom_from_fchk,
#     geom_from_hessian,
#     geom_from_mol2,
#     geom_from_pdb,
#     geom_from_qcschema,
#     save_hessian as save_h5_hessian,
#     geom_from_zmat_fn,
#     geoms_from_inline_xyz,
#     geom_from_pubchem_name,
# )


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

    atom_lines = lines[1:]
    atoms: List[Tuple[str, Tuple[float, float, float]]] = []
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if i == 0:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
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
    # mol = gto.Mole()
    # mol.atom = atoms  # list[(symbol, (x,y,z))]
    # mol.basis = "6-31g(d)"
    mol = pyscf.M(atom=atoms, basis="6-31g(d)")
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.unit = "Bohr"
    mol.build()
    return mol


def compute_hessian(
    mol: gto.Mole, multiplicity: int = 1, xc: str = "wb97x", debug_hint=""
) -> np.ndarray:
    is_open_shell = multiplicity != 1
    if pyscf_on_gpu:
        # restricted Kohn-Sham (RKS)
        # Density fitting (DF), sometimes also called the resolution of identity (RI) approximation,
        # is a method to approximate the four-index electron repulsion integrals (ERIs) by two- and three-index tensors
        mf = rks.RKS(mol, xc=xc).density_fit()
    else:
        if is_open_shell:
            mf = dft.UKS(mol)
        else:
            mf = dft.RKS(mol)

    mf.xc = xc
    # Tighten SCF a bit for stability
    # mf.conv_tol = 1e-12
    # mf.max_cycle = 400
    mf.verbose = 0  # Suppress SCF convergence messages
    mf.grids.level = 5
    try:
        _ = mf.kernel()  # compute total energy
        # g = mf.nuc_grad_method()
        # g_dft = g.kernel()   # compute analytical gradient
    except Exception as e:
        print("\n" + ">" * 40)
        print(f"Error in SCF: {debug_hint}: \n{e}")
        print("<" * 40)
        return None
    if not mf.converged:
        print("\n" + ">" * 40)
        print(f"SCF did not converge: {debug_hint}")
        print("<" * 40)
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
    atoms = read_xyz(xyz_path)  # Angstrom
    # convert Angstrom to Bohr
    atoms = [
        (sym, (x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR)) for sym, (x, y, z) in atoms
    ]
    mol = build_molecule(atoms)
    hessian_dft = compute_hessian(mol, xc="wb97x")
    return hessian_dft


def get_proposals_dir(inp_path, hessian_method=None):
    run_root = None
    proposals_dir = None
    if os.path.isdir(inp_path):
        # Accept either the proposals dir itself or the run root containing scratch
        if os.path.basename(inp_path) == "ts_proposal_geoms":
            proposals_dir = os.path.abspath(inp_path)
            run_root = os.path.dirname(inp_path)
        else:
            # try scratch/ts_proposal_geoms under this path
            cand = glob.glob(
                os.path.join(inp_path, "**/ts_proposal_geoms"), recursive=True
            )
            if len(cand) == 1:
                proposals_dir = os.path.abspath(cand[0])
                run_root = os.path.dirname(cand[0])
            elif os.path.isdir(os.path.join(inp_path, "ts_proposal_geoms")):
                proposals_dir = os.path.abspath(
                    os.path.join(inp_path, "ts_proposal_geoms")
                )
                run_root = os.path.abspath(inp_path)
    else:
        # assume run name is passed; resolve to ReactBench runs scratch dir used by main.py
        # First find ts_geoms_hessians to deduce run root
        print(f"\nLooking for {inp_path}*ts1x_{hessian_method}/ts_geoms_hessians")
        _paths = glob.glob(f"runs/{inp_path}*ts1x_{hessian_method}/ts_geoms_hessians")
        if len(_paths) == 1:
            run_root = os.path.dirname(_paths[0])
        else:
            raise ValueError(f"No or multiple runs found for {inp_path}: {_paths}")
        # proposals live next to ts_geoms_hessians in scratch
        proposals_dir = os.path.join(run_root, "ts_proposal_geoms")

    if proposals_dir is None or not os.path.isdir(proposals_dir):
        raise ValueError(f"Could not locate proposals directory: {proposals_dir}")
    return proposals_dir, run_root


def compare_proposals_with_dft(proposals_dir, run_root):
    runname = os.path.basename(run_root.rstrip("/"))
    outdir = f"runs_dft/{runname}"
    os.makedirs(outdir, exist_ok=True)

    # torch.manual_seed(42)
    np.random.seed(42)

    # Gather proposal xyz files: prefer one per rxn_ind, prioritize ts_opt over ts_final_geometry
    cand_xyz = sorted(glob.glob(os.path.join(proposals_dir, "*_ts_opt.xyz")))
    cand_xyz += sorted(
        glob.glob(os.path.join(proposals_dir, "*_ts_final_geometry.xyz"))
    )
    # Deduplicate by rxn_ind prefix before first underscore
    seen = set()
    proposal_xyz = []
    for p in cand_xyz:
        rxn_ind = os.path.basename(p).split("_")[0]
        if rxn_ind in seen:
            continue
        seen.add(rxn_ind)
        proposal_xyz.append(p)

    print(f"Found {len(proposal_xyz)} unique proposal geometries")

    # DFT evaluation: compute Hessian eigen-analysis and SCF gradient RMS
    rows = []
    cnt_ts = 0
    cnt_force_ok = 0
    cnt_ts_force_ok = 0
    force_rms_cutoff = 3.0e-3  # Hartree/Bohr default Gaussian RMS force

    # optionally subsample
    print()
    cnt_success_dft = 0
    for xyz_path in tqdm(
        proposal_xyz, desc="Verify proposals via DFT", total=args.max_samples
    ):
        if cnt_success_dft >= args.max_samples:
            break
        rxn_ind = os.path.basename(xyz_path).split("_")[0]
        h5_path = os.path.join(outdir, f"{rxn_ind}_proposal_hessian_dft.h5")

        # Build molecule in Bohr
        atoms = read_xyz(xyz_path)
        atoms_bohr = [
            (sym, (x * ANG2BOHR, y * ANG2BOHR, z * ANG2BOHR))
            for sym, (x, y, z) in atoms
        ]
        mol = build_molecule(atoms_bohr)

        atomsymbols = [a[0] for a in atoms]
        coords_bohr = np.array([a[1] for a in atoms_bohr]).reshape(-1, 3)
        cart_xyz = coords_bohr.reshape(-1)

        # Hessian: try to load cached, otherwise compute and save
        hess = None
        if os.path.exists(h5_path) and not args.redo:
            with h5py.File(h5_path, "r") as handle:
                _atoms = [
                    atom.capitalize().decode("utf-8") for atom in handle.attrs["atoms"]
                ]
                _coords3d = handle["coords3d"][:]  # Bohr
                _energy = handle.attrs["energy"]  # Hartree
                assert _atoms == atomsymbols, f"{_atoms} != {atomsymbols}"
                assert np.allclose(_coords3d, coords_bohr), (
                    f"{np.max(np.abs(_coords3d / coords_bohr)):.1e}"
                )
                hess = handle["hessian"][:]  # Hartree/Bohr^2
                if "forces" in handle:
                    _forces = handle["forces"][:]
                    force_rms = float(np.sqrt(np.mean(_forces**2)))
            tqdm.write(f"Loaded Hessian for {rxn_ind} from {h5_path}")
            tqdm.write(f"Force RMS: {force_rms:.1e} Ha/Bohr")

        # Compute Hessian if not found
        if hess is None:
            # SCF and gradient
            if pyscf_on_gpu:
                mf = rks.RKS(mol, xc="wb97x").density_fit()
            else:
                mf = dft.RKS(mol)
            mf.xc = "wb97x"
            mf.conv_tol = 1e-9
            mf.max_cycle = 200
            mf.verbose = 0
            _ = mf.kernel()
            grad = mf.Gradients().kernel()  # shape (N,3) Hartree/Bohr
            force_rms = float(np.sqrt(np.mean(grad**2)))
            print(f" SCF/Grad success for {rxn_ind}: {force_rms:.1e} Ha/Bohr")
            hess = compute_hessian(mol, xc="wb97x", debug_hint=rxn_ind)
            if hess is None:
                tqdm.write(f"Failed to compute Hessian for {rxn_ind}. Skipping...")
                continue
            _atomsymbols = [a[0] for a in atoms]
            _coords_bohr = np.array([a[1] for a in atoms_bohr]).reshape(-1, 3)
            with h5py.File(h5_path, "w") as handle:
                handle.attrs["atoms"] = np.array(
                    [a.capitalize() for a in _atomsymbols], dtype="S"
                )
                handle.create_dataset("coords3d", data=_coords_bohr)
                handle.create_dataset("hessian", data=hess)
                if grad is not None:
                    handle.create_dataset("forces", data=grad)
                handle.attrs["energy"] = 0.0
            tqdm.write(f"Saved Hessian for {rxn_ind} to {h5_path}")
        freqs = analyze_frequencies(hess, cart_xyz, atomsymbols)
        neg_num = int(freqs["neg_num"]) if freqs is not None else np.nan
        is_ts = bool(neg_num == 1)

        if is_ts:
            cnt_ts += 1
        if np.isfinite(force_rms) and force_rms <= force_rms_cutoff:
            cnt_force_ok += 1
            if is_ts:
                cnt_ts_force_ok += 1

        rows.append(
            {
                "rxn_ind": rxn_ind,
                "file": os.path.basename(xyz_path),
                "force_rms_Ha_per_Bohr": force_rms,
                "is_ts": is_ts,
                "neg_count": neg_num,
                "force_ok": force_rms <= force_rms_cutoff,
                "ts_force_ok": force_rms <= force_rms_cutoff and is_ts,
            }
        )
        cnt_success_dft += 1

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "verify_ts_with_dft_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSummary across {cnt_success_dft}/{len(proposal_xyz)} proposals")
    print(f"  TS count (1 imaginary mode): {cnt_ts}")
    print(f"  Force RMS <= {force_rms_cutoff:.1e} (Ha/Bohr): {cnt_force_ok}")
    print(f"  TS + Force RMS <= {force_rms_cutoff:.1e} (Ha/Bohr): {cnt_ts_force_ok}")
    # both = int(np.sum((df["is_ts"].astype(bool)) & (df["force_rms_Ha_per_Bohr"] <= force_rms_cutoff)))
    # print(f"  Both: {both}")
    print(f"Saved per-geometry results to: {csv_path}")
    return df, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF."
    )

    parser.add_argument(
        "inp_path",
        type=str,
        default="equiformer_hesspredhesspredalldatanumlayershessian3presetluca8w10onlybz128",
        help="Path to run directory or run name (used to resolve proposals).",
    )
    parser.add_argument(
        "--hessian_method",
        type=str,
        default="predict",
        help="Method tag used in run directory resolution (predict|autograd).",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the DFT Hessian computation. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process. If None, process all samples.",
    )
    parser.add_argument(
        "--force_rms_cutoff",
        type=float,
        default=3.0e-3,
        help="Force RMS cutoff in Hartree/Bohr. TS search used RMS force of 3.0e^-4",
    )

    args = parser.parse_args()

    if args.inp_path not in ["plot", None, "none", "None"]:
        proposals_dir, run_root = get_proposals_dir(args.inp_path, args.hessian_method)
        runname = os.path.basename(run_root.rstrip("/"))
        outdir = f"runs_dft/{runname}"
        print(f"Proposals dir: {proposals_dir}")
        print(f"Run root: {run_root}")

        df, csv_path = compare_proposals_with_dft(proposals_dir, run_root)

    else:
        dfs = []
        results_positive_rates = {}
        for inp_path, hessian_method, name in [
            ["equiformer_hesspred", "predict", "predict"],
            ["equiformer", "autograd", "autograd"],
        ]:
            print(f"\n# Processing {inp_path} with {hessian_method}")
            proposals_dir, run_root = get_proposals_dir(inp_path, hessian_method)
            runname = os.path.basename(run_root.rstrip("/"))
            outdir = f"runs_dft/{runname}"
            csv_path = os.path.join(outdir, "verify_ts_with_dft_summary.csv")
            df = pd.read_csv(csv_path)
            df["Method"] = name.capitalize()
            dfs.append(df)

            results_positive_rates[hessian_method] = {}

            #################################################################################################
            # Plot a histogram of the force RMS
            force_rms_cutoff = args.force_rms_cutoff
            outdir = os.path.dirname(csv_path)
            fig = px.histogram(
                df, x="force_rms_Ha_per_Bohr", nbins=100, title="Force RMS (Ha/Bohr)"
            )
            fig.add_vline(x=force_rms_cutoff, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title="Force RMS (Ha/Bohr)", yaxis_title="Count")
            fig_path = os.path.join(outdir, "force_rms_hist.png")
            fig.write_image(fig_path)
            print(f"Saved histogram to: {fig_path}")

            #################################################################################################
            # Compute a confusion matrix
            #################################################################################################
            """The idea.
            Each method proposes TS for each reaction or does not converge (force norm) / 
            discard guess due to frequency analysis.

            The more correct TS are proposed, the better the method.
            """

            force_rms_cutoff = args.force_rms_cutoff

            total = 960
            # positive = num proposed TS
            positive = len(glob.glob(os.path.join(proposals_dir, "*_ts_opt.xyz")))
            negative = total - positive
            false_negative = negative  # assume all samples should have a TS
            true_negative = 0
            print(f"Total: {total}")
            print(
                f"Positive: {positive} (Proposed TS: RS-RFO search converged and model Hessian has one imaginary frequency)"
            )
            print(f"Negative: {negative} (Proposed TS: workflow failed somewhere)")

            # sampled
            total_sampled = df.shape[0]
            positive_sampled = (
                total_sampled  # we are only checking positives (TS proposals)
            )
            for use_force_rms_threshold in [True, False]:
                print()
                if use_force_rms_threshold:
                    description = f"one negative eigenvalue and force RMS < {force_rms_cutoff:.1e} Ha/Bohr:"
                    print(description)
                    true_positive_sampled = df[
                        (df["is_ts"])
                        & (df["force_rms_Ha_per_Bohr"] <= force_rms_cutoff)
                    ].shape[0]
                else:
                    description = "one negative eigenvalue:"
                    print(description)
                    true_positive_sampled = df[(df["is_ts"])].shape[0]
                false_positive_sampled = total_sampled - true_positive_sampled

                true_positive_rate = true_positive_sampled / positive_sampled
                false_positive_rate = false_positive_sampled / positive_sampled
                true_negative_rate = 0.0
                false_negative_rate = 1.0

                correct_proposed_estimated = true_positive_rate * positive
                false_proposed_estimated = false_positive_rate * positive
                print(
                    f"True positive sampled: {true_positive_sampled} / {positive_sampled}"
                )
                print(f"Correct proposed estimated: {correct_proposed_estimated:.1f}")
                print(f"False proposed estimated: {false_proposed_estimated:.1f}")

                print(f"True Positive Rate: {true_positive_rate:.3f}")
                print(f"False Positive Rate: {false_positive_rate:.3f}")
                print(f"(True Negative Rate: {true_negative_rate:.3f})")
                print(f"(False Negative Rate: {false_negative_rate:.3f})")

                results_positive_rates[hessian_method][description] = {
                    "correct_proposed_estimated": correct_proposed_estimated,
                    "false_proposed_estimated": false_proposed_estimated,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                }

        #################################################################################################
        # Combined results
        #################################################################################################
        print("\n")
        df = pd.concat(dfs)
        # df.to_csv(os.path.join(outdir, "verify_ts_with_dft_summary_combined.csv"), index=False)
        # print(f"Saved combined results to: {os.path.join(outdir, 'verify_ts_with_dft_summary_combined.csv')}")

        outdir = "runs_dft/combined"
        os.makedirs(outdir, exist_ok=True)

        #################################################################################################
        # Plot a histogram of the force RMS
        #################################################################################################
        for _method in ["Predict", "Autograd"]:
            df_method = df[df["Method"] == _method]
            print(f"Found {df_method.shape[0]} geometries for {_method}")
        fig = px.histogram(
            df,
            x="force_rms_Ha_per_Bohr",
            color="Method",
            nbins=100,
            # title="Force RMS (Ha/Bohr)",
            histnorm="probability density",
            # barnorm='percent', # fraction, percent
            opacity=0.8,
            marginal="rug",  # can be rug, `box`, `violin`
            # barmode="overlay",
            # log_x=True, # does not work
            range_x=[0, None],
        )
        fig.add_vline(x=force_rms_cutoff, line_dash="dash", line_color="black")
        fig.update_layout(
            xaxis_title="Force RMS (Ha/Bohr)",
            yaxis_title="Density",
            xaxis_range=[0, None],
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                x=0.98,
                y=0.6,
                xanchor="right",
                yanchor="top",
                # bgcolor="rgba(255,255,255,0.6)",
                # bordercolor="rgba(0,0,0,0.2)",
                # borderwidth=1
            ),
        )
        fig_path = os.path.join(outdir, "force_rms_hist_combined.png")
        fig.write_image(fig_path)
        print(f"Saved histogram to: {fig_path}")

        #################################################################################################
        # KDE plot of the force RMS by method (see-through)
        #################################################################################################
        # Prepare data per method
        kde_groups = []
        kde_labels = []
        for _method in ["Predict", "Autograd"]:
            vals = (
                df[df["Method"] == _method]["force_rms_Ha_per_Bohr"]
                .dropna()
                .astype(float)
                .values.tolist()
            )
            if len(vals) > 0:
                kde_groups.append(vals)
                kde_labels.append(_method)
        if len(kde_groups) >= 1:
            colors = [px.colors.qualitative.Plotly[i] for i in range(len(kde_groups))]
            fig_kde = ff.create_distplot(
                kde_groups, kde_labels, show_hist=False, show_rug=False, colors=colors
            )
            for tr in fig_kde.data:
                tr.opacity = 0.6
            fig_kde.add_vline(x=force_rms_cutoff, line_dash="dash", line_color="black")
            fig_kde.update_layout(
                # title="Force RMS KDE (Ha/Bohr)",
                xaxis_title="Force RMS (Ha/Bohr)",
                yaxis_title="Density",
                xaxis_range=[0, 0.02],
                margin=dict(l=1, r=0, t=0, b=0),
                legend=dict(
                    x=0.98,
                    y=0.98,
                    xanchor="right",
                    yanchor="top",
                    # bgcolor="rgba(255,255,255,0.6)",
                    # bordercolor="rgba(0,0,0,0.2)",
                    # borderwidth=1
                ),
            )
            fig_kde_path = os.path.join(outdir, "force_rms_kde_combined.png")
            fig_kde.write_image(fig_kde_path)
            print(f"Saved KDE plot to: {fig_kde_path}")

        # save results_positive_rates to a json file
        _path = os.path.join(outdir, "results_positive_rates.json")
        with open(_path, "w") as f:
            json.dump(results_positive_rates, f)
        print(f"Saved results_positive_rates to: {_path}")
