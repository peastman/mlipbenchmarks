import sys
import time
from ase import Atoms
import ase.io
import ase.md
import ase.md.velocitydistribution
import ase.optimize
import ase.units
import torch
import models
import h5py
from openff.toolkit.topology import Molecule

bohr_to_A = 0.529177210903

name = sys.argv[1]
model = sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
file = h5py.File('spice/SPICE-test.hdf5')
group = file[name]
smiles = group['smiles'].asstr()[0]
mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
charge = int(mol.total_charge.m)
total_atomic_number = int(sum(atom.atomic_number for atom in mol.atoms))
spin = 1 if (charge+total_atomic_number) % 2 == 0 else 2
atoms = Atoms(numbers=group['atomic_numbers'])
initial_memory = torch.cuda.device_memory_used(0)
atoms.calc = models.create_calculator(model)
models.set_charge(atoms, model, charge, spin)
atoms.set_positions(group['conformations'][0]*bohr_to_A)
opt = ase.optimize.LBFGS(atoms)
print('Optimizing...')
t1 = time.time()
opt.run(steps=50)
t2 = time.time()
print('Optimization time:', t2-t1)
md = ase.md.Langevin(atoms, 1*ase.units.fs, temperature_K=300, friction=0.001/ase.units.fs)
ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)
print('Equilibrating...')
md.run(100)
print('Simulating...')
t1 = time.time()
md.run(steps)
t2 = time.time()
print('Simulation time:', t2-t1)
print('Steps/second:', steps/(t2-t1))
final_memory = torch.cuda.device_memory_used(0)
print('Memory (GB):', (final_memory-initial_memory)/2**30)
print(model)
