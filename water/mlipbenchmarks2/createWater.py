from openmm import *
from openmm.app import *

ff = ForceField('tip3pfb.xml')
for width in [2, 3, 4, 5, 6]:
    modeller = Modeller(Topology(), [])
    modeller.addSolvent(ff, boxSize=(width, width, width))
    system = ff.createSystem(modeller.topology, nonbondedMethod=PME)
    simulation = Simulation(modeller.topology, system, VerletIntegrator(0.001))
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    positions = simulation.context.getState(positions=True).getPositions()
    PDBFile.writeFile(modeller.topology, positions, f'water{width}.pdb')
