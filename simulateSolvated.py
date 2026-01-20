import sys
import ase.io
import ase.md
import ase.md.velocitydistribution
import ase.optimize
import ase.units
import models
import numpy as np
from rdkit import Chem

model = sys.argv[1]
steps = 10000
temperature = 400
filename = 'spice/solvated.pdb'
atoms = ase.io.read(filename)
atoms.calc = models.create_calculator(model)
atoms.info['charge'] = 0
atoms.info['spin'] = 1

mol = Chem.MolFromPDBFile(filename, removeHs=False)
bonds = []
for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    bonds.append((i, j, atoms.get_distance(i, j, True)))
for atom in mol.GetAtoms():
    if atom.GetPDBResidueInfo().GetName().strip() == 'O':
        # RDKit doesn't add bonds to water correctly.
        i = atom.GetIdx()
        bonds.append((i, i+1, atoms.get_distance(i, i+1, True)))
        bonds.append((i, i+2, atoms.get_distance(i, i+2, True)))

opt = ase.optimize.LBFGS(atoms)
print('Optimizing...')
opt.run(steps=50)
md = ase.md.Langevin(atoms, 1*ase.units.fs, temperature_K=temperature, friction=0.001/ase.units.fs)
ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
print('Equilibrating...')
md.run(1000)
print('Simulating...')
temps = []
for _ in range(steps):
    md.run(1)
    temps.append(atoms.get_temperature())
print('Average temperature:', np.mean(temps))
print('Standard deviation:', np.std(temps))
print('Maximum temperature:', np.max(temps))
error = False
for i, j, dist in bonds:
    dist2 = atoms.get_distance(i, j, True)
    if dist2-dist > 0.5:
        name1 = mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
        name2 = mol.GetAtomWithIdx(j).GetPDBResidueInfo().GetName().strip()
        print(i, j, name1, name2, dist, dist2)
        error = True
if error:
    ase.io.write(f'stability/{model}.pdb', atoms)
print(model)
