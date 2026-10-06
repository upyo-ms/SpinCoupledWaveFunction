#!/usr/bin/env python

'''
Write orbitals, electron density, molecular electrostatic potential in
Gaussian cube file format.
'''

from pyscf import gto, scf
from pyscf.tools import cubegen

import numpy as np

mol = gto.Mole()
mol.charge = 0
mol.spin = 0
mol.build(atom = 'H 0 0 0; Li 0 0 2.4', basis ='sto-3G')

C = np.asarray(
[[ 3.72467580e-01,  9.16607431e-01,  3.72467511e-01, -9.16607086e-01],
 [ 9.34779712e-01, -3.85210107e-01,  9.34779574e-01,  3.85210202e-01],
 [-3.63648921e-02,  4.53591913e-02, -3.63645829e-02, -4.53598836e-02],
 [-6.89637584e-03, -2.04748774e-02, -6.89730684e-03,  2.04758328e-02],
 [-5.22398802e-07,  2.39557399e-07, -2.46160163e-07,  1.97564937e-07],
 [-5.42172066e-07,  4.03689396e-07, -4.15709042e-07,  6.40194013e-08]])

gto.tofile(mol, 'hli_24.xyz')
cubegen.orbital(mol, 'hli_24_mo1.cube', C[:, 0])
cubegen.orbital(mol, 'hli_24_mo2.cube', C[:, 1])
cubegen.orbital(mol, 'hli_24_mo3.cube', C[:, 2])
cubegen.orbital(mol, 'hli_24_mo4.cube', C[:, 3])
