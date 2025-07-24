"""
This module contains miscellaneous functions borrowed from the taffi package that are
useful for yarp.
"""

import numpy as np
from scipy.spatial.distance import cdist
from copy import copy, deepcopy

from ReactBench.utils.properties import el_radii, el_max_bonds


def table_generator(elements, geometry, scale_factor=1.2, filename=None):
    """
    Algorithm for finding the adjacency matrix of a geometry based on atomic separations.

    Parameters
    ----------
    elements : list
               Contains elemental information indexed to the supplied adjacency matrix.
               Expects a list of lower-case elemental symbols.

    geo : array
          nx3 array of atomic coordinates (cartesian) in angstroms.

    scale_factor: float, default=1.2
                  Used to scale the atomic radii to determine if a bond exists.

    Returns
    -------
    adj_mat : array
              An nxn array indexed to elements containing ones bonds occur.
    """

    # Print warning for uncoded elements.
    for i in elements:
        if i not in el_radii.keys():
            print(
                "ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(
                    i
                )
                + " dictionary before proceeding. Exiting..."
            )
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    dist_mat = np.triu(cdist(geometry, geometry))

    # Find plausible connections
    x_ind, y_ind = np.where(
        (dist_mat > 0.0)
        & (dist_mat < max([el_radii[i] ** 2.0 for i in el_radii.keys()]))
    )

    # Initialize the adjacency matrix
    adj_mat = np.zeros([len(geometry), len(geometry)])

    # Iterate over plausible connections and determine actual connections
    for count, i in enumerate(x_ind):

        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if (
            dist_mat[i, y_ind[count]]
            < (el_radii[elements[i]] + el_radii[elements[y_ind[count]]]) * scale_factor
        ):
            adj_mat[i, y_ind[count]] = 1

        # Special treatment of hydrogens
        if elements[i] == "H" and elements[y_ind[count]] == "H":
            if (
                dist_mat[i, y_ind[count]]
                < (el_radii[elements[i]] + el_radii[elements[y_ind[count]]]) * 1.5
            ):
                adj_mat[i, y_ind[count]] = 1

    # Hermitize Adj_mat
    adj_mat = adj_mat + adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = {i: 0 for i in el_radii.keys()}
    for count_i, i in enumerate(adj_mat):

        if (
            el_max_bonds[elements[count_i]] is not None
            and sum(i) > el_max_bonds[elements[count_i]]
        ):
            problem_dict[elements[count_i]] += 1
            cons = sorted(
                [
                    (
                        (dist_mat[count_i, count_j], count_j)
                        if count_j > count_i
                        else (dist_mat[count_j, count_i], count_j)
                    )
                    for count_j, j in enumerate(i)
                    if j == 1
                ]
            )[::-1]
            while sum(adj_mat[count_i]) > el_max_bonds[elements[count_i]]:
                sep, idx = cons.pop(0)
                adj_mat[count_i, idx] = 0
                adj_mat[idx, count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum([problem_dict[i] for i in problem_dict.keys()]) > 0:
        print("Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if filename is None:
                    if i == "H":
                        print(
                            "WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "C":
                        print(
                            "WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "Si":
                        print(
                            "WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "F":
                        print(
                            "WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "Cl":
                        print(
                            "WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "Br":
                        print(
                            "WARNING in Table_generator: {} bromine(s) have more than one bond.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "I":
                        print(
                            "WARNING in Table_generator: {} iodine(s) have more than one bond.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "O":
                        print(
                            "WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "N":
                        print(
                            "WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(
                                problem_dict[i]
                            )
                        )
                    if i == "B":
                        print(
                            "WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(
                                problem_dict[i]
                            )
                        )
                else:
                    if i == "H":
                        print(
                            "WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "C":
                        print(
                            "WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "Si":
                        print(
                            "WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "F":
                        print(
                            "WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "Cl":
                        print(
                            "WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "Br":
                        print(
                            "WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "I":
                        print(
                            "WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "O":
                        print(
                            "WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "N":
                        print(
                            "WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(
                                filename, problem_dict[i]
                            )
                        )
                    if i == "B":
                        print(
                            "WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(
                                filename, problem_dict[i]
                            )
                        )
        print("")

    return adj_mat


# Return ring(s) that atom idx belongs to
# The algorithm spawns non-backtracking walks along the graph. If the walk encounters the starting node, that consistutes a cycle.
def return_ring_atoms(
    adj_list, idx, start=None, ring_size=10, counter=0, avoid_set=None, convert=True
):

    # Consistency/Termination checks
    if ring_size < 3:
        print(
            "ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!"
        )

    # Break if search has been exhausted
    if counter == ring_size:
        return []

    # Automatically assign start to the supplied idx value. For recursive calls this is updated each call
    if start is None:
        start = idx

    # Initially set to an empty set, during recursion this is occupied by already visited nodes
    if avoid_set is None:
        avoid_set = set([])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)

    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)

    # grab current connections while avoiding backtracking
    cons = adj_list[idx].difference(avoid_set)

    # You have run out of graph
    if len(cons) == 0:
        return []

    # You discovered the starting point
    elif start in cons:
        avoid_set.add(start)
        return [avoid_set]

    # The search continues
    else:
        rings = []
        for i in cons:
            rings = rings + [
                i
                for i in return_ring_atoms(
                    adj_list,
                    i,
                    start=start,
                    ring_size=ring_size,
                    counter=counter + 1,
                    avoid_set=copy(avoid_set),
                    convert=convert,
                )
                if i not in rings
            ]

    # Return of the original recursion is list of lists containing ordered atom indices for each cycle
    if counter == 0:
        if convert:
            return [ring_path(adj_list, _) for _ in rings]
        else:
            return rings

    # Return of the other recursions is a list of index sets for each cycle (sets are faster for comparisons)
    else:
        return rings


def return_rings(adj_list, max_size=20, remove_fused=True):
    """
    Finds the rings in a molecule based on its adjacency matrix. Most of the work in this function is done by
    `return_ring_atom()`, this function just cleans up the outputs and collates rings of different sizes.

    Parameters
    ----------
    adj_list: list of lists
              A sublist is contained for each atom holding the indices of its bonded neighbors.

    max_size: int, default=20
              Determines the maximum size of rings to return.

    remove_fused: bool, default=True
                  Controls whether fused rings are returned (False) or not (True).

    Returns
    -------
    rings: list
           List of lists holding the atom indices in each ring.
    """

    # Identify rings
    rings = []
    ring_size_list = range(max_size + 1)[3:]  # starts at 3
    for i in range(len(adj_list)):
        rings += [
            _
            for _ in return_ring_atoms(adj_list, i, ring_size=max_size, convert=False)
            if _ not in rings
        ]

    # Remove fused rings based on if another ring's atoms wholly intersect a given ring
    if remove_fused:
        del_ind = []
        for count_i, i in enumerate(rings):
            if count_i in del_ind:
                continue
            else:
                del_ind += [
                    count
                    for count, _ in enumerate(rings)
                    if count != count_i
                    and count not in del_ind
                    and i.intersection(_) == i
                ]
        del_ind = set(del_ind)

        # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring
        rings = [_ for count, _ in enumerate(rings) if count not in del_ind]

    # ring_path is used to convert the ring sets into actual ordered sets of indices that create the ring.
    # rings are sorted by size
    rings = sorted([ring_path(adj_list, _, path=None) for _ in rings], key=len)

    # Return list of rings or empty list
    if rings:
        return rings
    else:
        return []


# Convenience function for generating an ordered sequence of indices that enumerate a ring starting from the set of ring indices generated by return_ring_atoms()
def ring_path(adj_list, ring, path=None):

    # Initialize the loop starting from the minimum index, with the traversal direction set by min bonded index.
    if path is None:
        path = [min(ring), min(adj_list[min(ring)].intersection(ring))]

    # This for recursive construction is needed to handle branching possibilities. All branches are followed and only the one yielding the full cycle is returned
    for i in [_ for _ in adj_list[path[-1]] if _ in ring and _ not in path]:
        try:
            path = ring_path(adj_list, ring, path=path + [i])
            return path
        except:
            pass

    # Eventually the recursions will reach the end of a cycle (i.e., for i in []: for the above loop) and hit this.
    # If the path is shorter than the full cycle then it is invalid (i.e., the wrong branch was followed somewhere)
    if len(path) == len(ring):
        return path
    else:
        raise Exception(
            "wrong path, didn't recover ring"
        )  # This never gets printed, it is just used to trigger the except at a higher level of recursion.


# Convenience function for converting between adjacency matrix and adjacency list (actually a list of sets for convenience)
def adjmat_to_adjlist(adj_mat):
    return [set(np.where(_ == 1)[0]) for _ in adj_mat]


def graph_seps(adj_mat_0):
    """
    Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix

    Parameters
    ----------
    adj_mat_0 : array
            This array is indexed to the atoms in the `yarpecule` and has a one at row i and column j if there is
            a bond (of any kind) between the i-th and j-th atoms.

    Returns
    ----------
    seps : NDArray
            What is the final shape of this matrix? (ERM)
    """

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)

    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat), len(adj_mat)]) * -1
    np.fill_diagonal(seps, 0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):

        # All perform assignments to unassigned elements (seps==-1)
        # and all perform an assignment if the value in the adj_mat is > 0
        seps[np.where((seps == -1) & (adj_mat > 0))] = i + 1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat > 1)] = 1

        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat, adj_mat_0)

    return seps


# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2.
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, point the rotation passes through
# v2: 1x3 array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)


# Simply convert bond matrix into adjacency matrix
def bondmat_to_adjmat(bond_mat):

    # Create a matrix with all zeros and set the diagonal elements to 0
    adj_mat = np.zeros((bond_mat.shape[0], bond_mat.shape[0]))

    # Set off-diagonal elements to 1 where corresponding mat1 elements > 0
    adj_mat[bond_mat > 0] = 1
    np.fill_diagonal(adj_mat, 0)

    return adj_mat
