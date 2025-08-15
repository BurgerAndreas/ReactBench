#!/usr/bin/env python3
"""
Compute the nuclear Hessian from an input XYZ geometry using DFT
at the ωB97X/6-31G(d) level of theory.

Requirements:
- pyscf (pip install pyscf)

Usage:
  python scripts/compute_hessian_dft.py input.xyz --charge 0 --mult 1 \
         --out hessian.npy --txt hessian.txt

Notes:
- The Hessian is saved in atomic units (Hartree/Bohr^2) as a NumPy array.
- The XYZ coordinates are interpreted as Angstrom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np

from pyscf import dft, gto


def read_xyz(path: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    try:
        nat = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError(f"Invalid XYZ header in {path}: {exc}") from exc

    atom_lines = lines[2 : 2 + nat]
    atoms: List[Tuple[str, Tuple[float, float, float]]] = []
    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: '{ln}'")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception as exc:
            raise ValueError(f"Invalid coordinates in line: '{ln}'") from exc
        atoms.append((sym, (x, y, z)))
    if len(atoms) != nat:
        raise ValueError(
            f"XYZ atom count mismatch: header says {nat}, parsed {len(atoms)}"
        )
    return atoms


def build_molecule(
    atoms: List[Tuple[str, Tuple[float, float, float]]], charge: int, multiplicity: int
) -> gto.Mole:
    spin = multiplicity - 1  # 2S = multiplicity - 1
    mol = gto.Mole()
    mol.atom = atoms  # list[(symbol, (x,y,z))]
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = "6-31g(d)"
    mol.unit = "Angstrom"
    mol.build()
    return mol


def compute_hessian(mol: gto.Mole, multiplicity: int, xc: str = "wb97x") -> np.ndarray:
    is_open_shell = multiplicity != 1
    start_time = time.time()
    if is_open_shell:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    # Tighten SCF a bit for stability
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    print(f"SCF time: {time.time() - start_time:.1f} seconds")
    if not mf.converged:
        raise RuntimeError("SCF did not converge; aborting Hessian computation.")

    # Use the generic interface available on the SCF object
    # (N, N, 3, 3) where N is number of atoms
    hessian = mf.Hessian().kernel()
    print(f"Hessian time: {time.time() - start_time:.1f} seconds")

    # Properly reshape from (N, N, 3, 3) to (3N, 3N)
    # Need to transpose axes to get proper ordering: [atom_i, coord_i, atom_j, coord_j]
    N = mol.natm
    hes = hessian.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    print(f"Symmetry error: {np.sum(np.abs(hes - hes.T))}")

    # hes shape: (3N, 3N), atomic units (Hartree/Bohr^2)
    return hes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DFT Hessian at ωB97X/6-31G(d) from an XYZ geometry using PySCF."
        )
    )
    parser.add_argument("xyz", help="Path to input XYZ file")
    parser.add_argument(
        "--charge", type=int, default=0, help="Total charge (default: 0)"
    )
    parser.add_argument(
        "--mult", type=int, default=1, help="Spin multiplicity (default: 1)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npy file for Hessian (default: <xyz_basename>_hessian.npy)",
    )
    parser.add_argument(
        "--txt",
        type=str,
        default=None,
        help="Optional text dump of Hessian (if provided)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional JSON file to save run metadata",
    )
    args = parser.parse_args()

    xyz_path = os.path.abspath(args.xyz)
    atoms = read_xyz(xyz_path)
    n_atoms = len(atoms)

    mol = build_molecule(atoms, charge=args.charge, multiplicity=args.mult)
    hes = compute_hessian(mol, multiplicity=args.mult, xc="wb97x")

    if args.out is None:
        base = os.path.splitext(os.path.basename(xyz_path))[0]
        out_path = os.path.abspath(f"{base}_hessian.npy")
    else:
        out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.save(out_path, hes)

    if args.txt:
        txt_path = os.path.abspath(args.txt)
        np.savetxt(txt_path, hes)

    if args.metadata:
        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "xyz": xyz_path,
            "n_atoms": n_atoms,
            "charge": args.charge,
            "multiplicity": args.mult,
            "method": "DFT",
            "functional": "ωB97X (PySCF xc='wb97x')",
            "basis": "6-31G(d)",
            "units": {
                "coordinates": "Angstrom",
                "hessian": "Hartree/Bohr^2",
            },
            "output_npy": out_path,
            "output_txt": os.path.abspath(args.txt) if args.txt else None,
            "pyscf_version": getattr(gto, "__version__", None),
        }
        with open(os.path.abspath(args.metadata), "w") as f:
            json.dump(meta, f, indent=2)

    print(
        f"Hessian computed for {n_atoms} atoms. Saved to: {out_path}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
