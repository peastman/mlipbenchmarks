import h5py
from ase import Atoms
import ase.io
import models
from openff.toolkit.topology import Molecule
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

bohr_to_A = 0.529177210903
hartree_to_kJpermole = 2625.4996394798254
ev_to_kJpermole = 96.48533212331002

model = sys.argv[1]
calc = models.create_calculator(model)
file = h5py.File('spice/SPICE-test.hdf5')
elements = models.supported_elements(model)
names = []
sizes = []
charges = []
errors = []
for name in file:
    group = file[name]
    if any(n not in elements for n in group['atomic_numbers']):
        continue
    smiles = group['smiles'].asstr()[0]
    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    charge = int(mol.total_charge.m)
    total_atomic_number = int(sum(atom.atomic_number for atom in mol.atoms))
    spin = 1 if (charge+total_atomic_number) % 2 == 0 else 2
    atoms = Atoms(numbers=group['atomic_numbers'])
    atoms.calc = calc
    models.set_charge(atoms, model, charge, spin)
    num_atoms = len(group['atomic_numbers'])
    energies = []
    for positions, energy in zip(group['conformations'], group['formation_energy']):
        atoms.set_positions(positions*bohr_to_A)
        ref_energy = energy*hartree_to_kJpermole
        energy = atoms.get_potential_energy()*ev_to_kJpermole
        energies.append((ref_energy, energy))
    error = 0.0
    n = len(energies)
    for i in range(n):
        for j in range(i):
            error += abs((energies[i][0]-energies[j][0]) - (energies[i][1]-energies[j][1]))
    error /= n*(n-1)/2
    names.append(name)
    sizes.append(num_atoms)
    charges.append(charge)
    errors.append(error)

df = pd.DataFrame({
    'name': names,
    'atoms': sizes,
    'charge': charges,
    'error': errors
})
df.to_csv(f'spice/{model}.csv')
print('Total Error:', np.mean(errors))

