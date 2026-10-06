"""Optional geometry-aware updates in the unchanged SCGVB variational space."""
import numpy as np
from scipy.linalg import eigh


def projected_spin_solve(system, C, incoming, cutoff):
    """Whiten actual CI vectors and project (H-E_reference), not large H/S entries.

    The incoming physical state is always part of the trial space. Determinant
    amplitudes remain derived from the orbitals and spin coefficients.
    """
    incoming = np.asarray(incoming, dtype=float)
    incoming = incoming/np.linalg.norm(incoming)
    metric = system.spin_overlap_matrix(C)
    metric = .5*(metric+metric.T)
    values, vectors = eigh(metric)
    scale = values[-1]
    if (not np.isfinite(values).all() or scale <= 0 or values[0] < -1e-8*scale):
        raise FloatingPointError('Invalid multi-electron spin overlap')
    keep = values > cutoff*scale
    numerator, norm = system.calcEnergy(C, incoming)
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Incoming spin state has no positive norm')
    reference = float(numerator/norm)
    basis = np.column_stack((vectors[:, keep]/np.sqrt(values[keep]), incoming/np.sqrt(norm)))
    shifted_h, physical_metric = system.projected_spin_matrices(C, basis, reference)
    small_values, small_vectors = eigh(.5*(physical_metric+physical_metric.T))
    if small_values[-1] <= 0 or small_values[0] < -1e-8*small_values[-1]:
        raise FloatingPointError('Invalid projected physical spin metric')
    resolved = small_values > max(1e-12, 100*np.finfo(float).eps*len(small_values))*small_values[-1]
    transform = small_vectors[:, resolved]/np.sqrt(small_values[resolved])
    reduced = np.einsum('ia,ij,jb->ab', transform, shifted_h, transform)
    energies, states = eigh(.5*(reduced+reduced.T))
    cs = np.einsum('ik,ka,a->i', basis, transform, states[:, 0])
    if not np.isfinite(cs).all() or np.linalg.norm(cs) == 0:
        raise FloatingPointError('Invalid projected spin solution')
    cs /= np.linalg.norm(cs)
    info = dict(spin_solver='projected', spin_energy_hartree=float(reference+energies[0]),
                spin_overlap_rank=int(keep.sum()), spin_functions=len(values),
                spin_subspace_dimension=int(resolved.sum()), spin_incoming_preserved=True,
                spin_overlap_cutoff=float(cutoff), spin_overlap_eigenvalues=values.tolist(),
                spin_projected_residual_max_abs=float(np.max(np.abs(
                    reduced@states[:, 0]-energies[0]*states[:, 0]))),
                spin_projected_metric_condition=float(small_values[-1]/small_values[resolved][0]))
    return cs, info


def orbital_geometry(system, C, Cs, root, invroot, relative_damping=1e-8):
    """Damped inverse metric on unit-column tangent coordinates.

    The metric measures the change of the normalized physical wavefunction.
    Damping regularizes the search direction, not the physical energy/model.
    """
    n, ne = C.shape
    metric = system.orbital_wavefunction_metric(C, Cs)
    transform = np.kron(invroot, np.eye(ne))
    d = np.einsum('ij,ja->ia', root, C)
    radial = np.zeros((C.size, ne))
    for j in range(ne):
        radial[np.arange(n)*ne+j, j] = d[:, j]
    projector = np.eye(C.size)-np.einsum('ij,kj->ik', radial, radial)
    transform = np.einsum('ij,jk->ik', transform, projector)
    metric = np.einsum('ia,ij,jb->ab', transform, metric, transform, optimize=True)
    metric = .5*(metric+metric.T)
    values, vectors = eigh(metric)
    if (not np.isfinite(values).all() or values[-1] <= 0
            or values[0] < -1e-7*values[-1]):
        raise FloatingPointError('Invalid orbital wavefunction metric')
    values = np.maximum(values, 0.)
    damping = max(1e-12, relative_damping*values[-1])
    inverse = np.einsum('ik,k,jk->ij', vectors, 1/(values+damping), vectors)
    inverse = np.einsum('ij,jk,kl->il', projector, inverse, projector, optimize=True)
    return inverse, metric, dict(orbital_metric_damping=float(damping),
                                orbital_metric_max_eigenvalue=float(values[-1]),
                                orbital_metric_relative_damping=relative_damping)
