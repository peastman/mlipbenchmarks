import h5py
from ase import Atoms
import ase.io
import models
from openff.toolkit.topology import Molecule
import sys
from collections import defaultdict
import numpy as np

bohr_to_A = 0.529177210903
bohr_to_nm = 0.0529177210903
A_to_nm = 0.1
hartree_to_kJpermole = 2625.4996394798254
ev_to_kJpermole = 96.48533212331002

model = sys.argv[1]
calc = models.create_calculator(model)
file = h5py.File('spice/SPICE-test.hdf5')
count_charged = defaultdict(int)
count_uncharged = defaultdict(int)
error_charged = defaultdict(float)
error_uncharged = defaultdict(float)
elements = models.supported_elements(model)
for name in file:
    group = file[name]
    smiles = group['smiles'].asstr()[0]
    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    charge = mol.total_charge.m
    if any(n not in elements for n in group['atomic_numbers']):
        continue
    total_atomic_number = sum(atom.atomic_number for atom in mol.atoms)
    spin = 1 if (charge+total_atomic_number) % 2 == 0 else 2
    atoms = Atoms(numbers=group['atomic_numbers'])
    atoms.calc = calc
    atoms.info['charge'] = charge
    atoms.info['spin'] = spin
    num_atoms = len(group['atomic_numbers'])
    for positions, grad in zip(group['conformations'], group['dft_total_gradient']):
        atoms.set_positions(positions*bohr_to_A)
        ref_forces = -grad*hartree_to_kJpermole/bohr_to_nm
        forces = atoms.get_forces()*ev_to_kJpermole/A_to_nm
        delta = forces-ref_forces
        error = np.sum(np.linalg.norm(delta, axis=1)/np.linalg.norm(ref_forces, axis=1))
        if charge == 0:
            count_uncharged[num_atoms] += num_atoms
            error_uncharged[num_atoms] += error
        else:
            count_charged[num_atoms] += num_atoms
            error_charged[num_atoms] += error
print('Charged:')
for num_atoms in sorted(count_charged.keys()):
    print(num_atoms, error_charged[num_atoms]/count_charged[num_atoms])
print('Uncharged:')
for num_atoms in sorted(count_uncharged.keys()):
    print(num_atoms, error_uncharged[num_atoms]/count_uncharged[num_atoms])
print('Total Error:')
print('charged:', sum(error_charged.values())/sum(count_charged.values()))
print('uncharged:', sum(error_uncharged.values())/sum(count_uncharged.values()))
print('combined:', (sum(error_charged.values())+sum(error_uncharged.values()))/(sum(count_charged.values())+sum(count_uncharged.values())))

