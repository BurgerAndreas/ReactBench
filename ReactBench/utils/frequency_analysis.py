import numpy as np
import scipy.constants as spc
import h5py

from ReactBench.utils.masses import MASS_DICT

"""
Adapted from 
dependencies/pysisyphus/pysisyphus/Geometry.py
"""
# from pysisyphus.constants import AU2J, BOHR2ANG, C, R, AU2KJPERMOL, NA
# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Å -> Bohr conversion factor
ANG2BOHR = 1 / BOHR2ANG
# Hartree to J
AU2J = spc.value("Hartree energy")
# Speed of light in m/s
C = spc.c
NA = spc.Avogadro


def inertia_tensor(coords3d, masses):
    """Inertita tensor.

                          | x² xy xz |
    (x y z)^T . (x y z) = | xy y² yz |
                          | xz yz z² |
    """
    x, y, z = coords3d.T
    squares = np.sum(coords3d**2 * masses[:, None], axis=0)
    I_xx = squares[1] + squares[2]
    I_yy = squares[0] + squares[2]
    I_zz = squares[0] + squares[1]
    I_xy = -np.sum(masses * x * y)
    I_xz = -np.sum(masses * x * z)
    I_yz = -np.sum(masses * y * z)
    return np.array(((I_xx, I_xy, I_xz), (I_xy, I_yy, I_yz), (I_xz, I_yz, I_zz)))


def get_trans_rot_vectors(cart_coords, masses, rot_thresh=1e-6):
    """Vectors describing translation and rotation.

    These vectors are used for the Eckart projection by constructing
    a projector from them.

    See Martin J. Field - A Pratcial Introduction to the simulation
    of Molecular Systems, 2007, Cambridge University Press, Eq. (8.23),
    (8.24) and (8.26) for the actual projection.

    See also https://chemistry.stackexchange.com/a/74923.

    Parameters
    ----------
    cart_coords : np.array, 1d, shape (3 * atoms.size, )
        Atomic masses in amu.
    masses : iterable, 1d, shape (atoms.size, )
        Atomic masses in amu.

    Returns
    -------
    ortho_vecs : np.array(6, 3*atoms.size)
        2d array containing row vectors describing translations
        and rotations.
    """

    coords3d = np.reshape(cart_coords, (-1, 3))
    total_mass = masses.sum()
    com = 1 / total_mass * np.sum(coords3d * masses[:, None], axis=0)
    coords3d_centered = coords3d - com[None, :]

    _, Iv = np.linalg.eigh(inertia_tensor(coords3d, masses))
    Iv = Iv.T

    masses_rep = np.repeat(masses, 3)
    sqrt_masses = np.sqrt(masses_rep)
    num = len(masses)

    def get_trans_vecs():
        """Mass-weighted unit vectors of the three cartesian axes."""

        for vec in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            _ = sqrt_masses * np.tile(vec, num)
            yield _ / np.linalg.norm(_)

    def get_rot_vecs():
        """As done in geomeTRIC."""

        rot_vecs = np.zeros((3, cart_coords.size))
        # p_vecs = Iv.dot(coords3d_centered.T).T
        for i in range(masses.size):
            p_vec = Iv.dot(coords3d_centered[i])
            for ix in range(3):
                rot_vecs[0, 3 * i + ix] = Iv[2, ix] * p_vec[1] - Iv[1, ix] * p_vec[2]
                rot_vecs[1, 3 * i + ix] = Iv[2, ix] * p_vec[0] - Iv[0, ix] * p_vec[2]
                rot_vecs[2, 3 * i + ix] = Iv[0, ix] * p_vec[1] - Iv[1, ix] * p_vec[0]
        rot_vecs *= sqrt_masses[None, :]
        return rot_vecs

    trans_vecs = list(get_trans_vecs())
    rot_vecs = np.array(get_rot_vecs())
    # Drop vectors with vanishing norms
    rot_vecs = rot_vecs[np.linalg.norm(rot_vecs, axis=1) > rot_thresh]
    tr_vecs = np.concatenate((trans_vecs, rot_vecs), axis=0)
    tr_vecs = np.linalg.qr(tr_vecs.T)[0].T
    return tr_vecs


def get_trans_rot_projector(cart_coords, masses, full=False):
    tr_vecs = get_trans_rot_vectors(cart_coords, masses=masses)
    U, s, _ = np.linalg.svd(tr_vecs.T)
    if full:
        P = np.eye(cart_coords.size)
        for tr_vec in tr_vecs:
            P -= np.outer(tr_vec, tr_vec)
    else:
        P = U[:, s.size :].T
    return P


def mass_weigh_hessian(hessian, masses3d):
    """mass-weighted hessian M^(-1/2) H M^(-1/2)
    Inverted square root of the mass matrix."""
    mm_sqrt_inv = np.diag(1 / (masses3d**0.5))
    return mm_sqrt_inv.dot(hessian).dot(mm_sqrt_inv)


def unweight_mw_hessian(mw_hessian, masses3d):
    """Unweight a mass-weighted hessian.
    Mass-weighted hessian to be unweighted
    ->
    2d array containing the hessian.
    """
    mm_sqrt = np.diag(masses3d**0.5)
    return mm_sqrt.dot(mw_hessian).dot(mm_sqrt)


def eckart_projection_notmw(hessian, cart_coords, atomsymbols):
    """Do Eckart projection starting from not-mass-weighted Hessian."""
    masses = np.array([MASS_DICT[atom.lower()] for atom in atomsymbols])
    masses3d = np.repeat(masses, 3)
    mw_hessian = mass_weigh_hessian(hessian, masses3d)
    P = get_trans_rot_projector(cart_coords, masses=masses, full=False)
    proj_hessian = P.dot(mw_hessian).dot(P.T)
    # Projection seems to slightly break symmetry (sometimes?). Resymmetrize.
    return (proj_hessian + proj_hessian.T) / 2

def eckart_projection_mw(mw_hessian, cart_coords, atomsymbols):
    """Do Eckart projection starting from mass-weighted Hessian."""
    masses = np.array([MASS_DICT[atom.lower()] for atom in atomsymbols])
    P = get_trans_rot_projector(cart_coords, masses=masses, full=False)
    proj_hessian = P.dot(mw_hessian).dot(P.T)
    # Projection seems to slightly break symmetry (sometimes?). Resymmetrize.
    return (proj_hessian + proj_hessian.T) / 2


def eigval_to_wavenumber(ev):
    # This approach seems numerically more unstable
    # conv = AU2J / (AMU2KG * BOHR2M ** 2) / (2 * np.pi * 3e10)**2
    # w2nu = np.sign(ev) * np.sqrt(np.abs(ev) * conv)
    # The two lines below are adopted from Psi4 and seem more stable,
    # compared to the approach above.
    conv = np.sqrt(NA * AU2J * 1.0e19) / (2 * np.pi * C * BOHR2ANG)
    w2nu = np.sign(ev) * np.sqrt(np.abs(ev)) * conv
    return w2nu

def load_hessian_h5(h5_path):
    with h5py.File(h5_path, "r") as handle:
        atoms = [atom.capitalize() for atom in handle.attrs["atoms"]]
        coords3d = handle["coords3d"][:] # Bohr
        energy = handle.attrs["energy"] # Hartree
        cart_hessian = handle["hessian"][:] # Hartree/Bohr^2
    return cart_hessian, atoms, coords3d, energy

def analyze_frequencies(
    hessian: np.ndarray | str, # Hartree/Bohr^2
    cart_coords: np.ndarray, # Bohr
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    if isinstance(hessian, str):
        _file = hessian
        hessian, atoms, coords3d, energy = load_hessian_h5(hessian) # Bohr and Hartree/Bohr^2
        # geom: Geometry = geom_from_hessian(hessian) # Bohr and Hartree/Bohr^2
        # assert np.allclose(geom.coords3d, cart_coords)
        assert np.allclose(cart_coords, coords3d), \
            f"XYZ and Hessian coordinates do not match\n {np.abs(cart_coords - coords3d).max():.1e}\n {np.abs(cart_coords - (coords3d / ANG2BOHR)).max():.1e}\n {_file}"
    
    proj_hessian = eckart_projection_notmw(hessian, cart_coords, atomsymbols)
    eigvals, eigvecs = np.linalg.eigh(proj_hessian)
    sorted_inds = np.argsort(eigvals)
    eigvals = eigvals[sorted_inds]
    eigvecs = eigvecs[:, sorted_inds]

    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    # eigval_str = np.array2string(eigvals[:10], precision=4)
    if neg_num > 0:
        wavenumbers = eigval_to_wavenumber(neg_eigvals)
        # wavenum_str = np.array2string(wavenumbers, precision=2)
    else:
        wavenumbers = None
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "wavenumbers": wavenumbers,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }
