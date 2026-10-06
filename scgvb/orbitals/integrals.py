from pyscf import scf, fci


def GetMolecule(mol, compute_fci=True):
    Ne = mol.tot_electrons()  # the number of electrons
    Sm = mol.spin+1  # Spin multiplicity

    # Calculationg the one and two electron intergrals between primitive basis functions
    # Use the molecule's spatial AO basis, including its spherical/cartesian
    # convention. Taking every second spinor component is not this basis.
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    JK = mol.intor('int2e')  # chemists' ordering: (ij|kl)

    Hcore = T+V

    e_fci = None
    if compute_fci:
        mf = scf.RHF(mol).run()
        cisolver = fci.FCI(mol, mf.mo_coeff)
        e_fci = cisolver.kernel()[0]

    return mol.energy_nuc(), Ne, Sm, S, Hcore, JK, e_fci
