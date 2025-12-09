import sys
import time
import ase.io
import ase.md
import ase.md.velocitydistribution
import ase.optimize
import ase.units
import torch

width = sys.argv[1]
model = sys.argv[2]
steps = int(sys.argv[3])
atoms = ase.io.read(f'water{width}.pdb')
initial_memory = torch.cuda.device_memory_used(0)
print(initial_memory)
match model:
    case 'mace-off23-small':
        from mace.calculators.foundations_models import mace_off
        atoms.calc = mace_off('small', default_dtype='float32')
    case 'mace-off24-medium':
        from mace.calculators.foundations_models import mace_off
        atoms.calc = mace_off('https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', default_dtype='float32')
    case 'mace-off23-large':
        from mace.calculators.foundations_models import mace_off
        atoms.calc = mace_off('large', default_dtype='float32')
    case 'mace-omol-0':
        from mace.calculators.foundations_models import mace_omol
        atoms.calc = mace_omol('extra_large', default_dtype='float32')
    case 'mace-mh-1':
        from mace.calculators.foundations_models import mace_mp
        atoms.calc = mace_mp('https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model', default_dtype='float32', head='spice_wB97M')
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
