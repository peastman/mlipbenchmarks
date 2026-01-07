from openmm import *
from openmm.app import *
from openmm.unit import *
import h5py
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

file = h5py.File('../spice/SPICE-test.hdf5')
group = file['WP0']
smiles = group['smiles'].asstr()[0]
mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
ff = ForceField('tip3pfb.xml')
smirnoff = SMIRNOFFTemplateGenerator(molecules=mol)
ff.registerTemplateGenerator(smirnoff.generator)
modeller = Modeller(mol.to_topology().to_openmm(), group['conformations'][0]*bohr)
modeller.addSolvent(ff, boxSize=(2, 2, 2))
system = ff.createSystem(modeller.topology, nonbondedMethod=PME)
simulation = Simulation(modeller.topology, system, LangevinIntegrator(300, 1.0, 0.001))
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
simulation.step(10000)
positions = simulation.context.getState(positions=True).getPositions()
PDBFile.writeFile(modeller.topology, positions, 'solvated.pdb')
