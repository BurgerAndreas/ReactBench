from ReactBench.utils.find_lewis import return_formals
from ReactBench.utils.taffi_functions import bondmat_to_adjmat
import numpy as np


def to_xyz_string(elements, geo, comment=""):
    """
    Simple wrapper function for creating xyz string

    Inputs     elements: list of element types (list of strings)
               geo:      Nx3 array holding the cartesian coordinates of the
                         geometry (atoms are indexed to the elements in Elements)

    Returns    string
    """
    xyz_string = f"{len(elements)}\n{comment}\n"
    for count_i, i in enumerate(elements):
        xyz_string += f"{i:<20s} {geo[count_i][0]:< 20.8f} {geo[count_i][1]:< 20.8f} {geo[count_i][2]:< 20.8f}\n"
    return xyz_string


def xyz_write(name, elements, geo, append_opt=False, comment=""):
    """
    Simple wrapper function for writing xyz file

    Inputs      name:     string holding the filename of the output
               elements: list of element types (list of strings)
               geo:      Nx3 array holding the cartesian coordinates of the
                         geometry (atoms are indexed to the elements in Elements)

    Returns     None
    """

    if append_opt == True:
        open_cond = "a"
    else:
        open_cond = "w"

    with open(name, open_cond) as f:
        f.write("{}\n".format(len(elements)))
        f.write("{}\n".format(comment))
        for count_i, i in enumerate(elements):
            f.write(
                "{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(
                    i, geo[count_i][0], geo[count_i][1], geo[count_i][2]
                )
            )
    return


def mol_write(name, elements, geo, bond_mat, q=0, append_opt=False):
    """
    Simple wrapper function for writing a mol (V2000) file

    Inputs      name:     string holding the filename of the output
                elements: list of element types (list of strings)
                geo:      Nx3 array holding the cartesian coordinates of the
                          geometry (atoms are indexed to the elements in Elements)
                bond_mat:  NxN array holding the molecular graph and lewis structure

    Returns     None
    """
    # Consistency check
    if len(elements) >= 1000:
        print(
            "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule."
        )
        return

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = "a"
    else:
        open_cond = "w"

    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    # only keep the file name
    if "/" in base_name:
        base_name = base_name.split("/")[-1]

    # obtain formal charge
    fc = list(return_formals(bond_mat, elements))

    # deal with radicals
    keep_lone = [count_i for count_i, i in enumerate(bond_mat) if i[count_i] % 2 == 1]

    # deal with charges
    chrg = len([i for i in fc if i != 0])

    # convert bond matrix to adj mat
    adj_mat = bondmat_to_adjmat(bond_mat)

    # Write the file
    with open(name, open_cond) as f:
        # Write the header
        f.write("{}\n     RDKit          3D\n\n".format(base_name))

        # Write the number of atoms and bonds
        f.write(
            "{:>3d}{:>3d}  0  0  1  0  0  0  0  0999 V2000\n".format(
                len(elements), int(np.sum(adj_mat / 2.0))
            )
        )

        # Write the geometry
        for count_i, i in enumerate(elements):
            f.write(
                " {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(
                    geo[count_i][0], geo[count_i][1], geo[count_i][2], i
                )
            )

        # Write the bonds
        bonds = [
            (count_i, count_j)
            for count_i, i in enumerate(adj_mat)
            for count_j, j in enumerate(i)
            if j == 1 and count_j > count_i
        ]
        for i in bonds:
            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0], i[1]])

            f.write(
                "{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(
                    i[0] + 1, i[1] + 1, bond_order
                )
            )

        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1, keep_lone[0] + 1, 2))
            elif len(keep_lone) == 2:
                f.write(
                    "M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(
                        2, keep_lone[0] + 1, 2, keep_lone[1] + 1, 2
                    )
                )
            else:
                keep_lone = []
                # print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write(
                    "M  CHG{:>3d}{:>4d}{:>4d}\n".format(
                        1, fc.index(charge) + 1, int(charge)
                    )
                )
            else:
                info = "M  CHG{:>3d}".format(chrg)
                for count_c, charge in enumerate(fc):
                    if charge != 0:
                        info += "{:>4d}{:>4d}".format(int(count_c) + 1, int(charge))
                info += "\n"
                f.write(info)

        f.write("M  END\n$$$$\n")

    return


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
