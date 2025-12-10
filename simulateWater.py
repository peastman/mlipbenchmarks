import sys
import time
import ase.io
import ase.md
import ase.md.velocitydistribution
import ase.optimize
import ase.units
import torch
import models

width = sys.argv[1]
model = sys.argv[2]
steps = int(sys.argv[3])
atoms = ase.io.read(f'water/water{width}.pdb')
initial_memory = torch.cuda.device_memory_used(0)
atoms.calc = models.create_calculator(model)
atoms.info['charge'] = 0
atoms.info['spin'] = 1
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
