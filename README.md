# MLIP Benchmarks

This repository contains scripts and data files from the paper

Peter Eastman, Evan Pretti, Thomas E. Markland.  "A Comparative Benchmark of Pretrained Machine Learning Interatomic
Potentials for Molecular Simulations."

### computeSpice.py

This script computes the accuracy of a model on the SPICE test set.  Run it as

```
python computeSpice.py <model>
```

where `<model>` is one of the model names defined in `models.py`.  It calculates the MAE for every molecule or dimer and
saves  the results to a file in the `spice` directory.

### simulateSpice.py

This script runs a short simulation of a molecule from the SPICE test set and reports the speed and memory use.
Run it as

```
python simulateSpice.py <molecule> <model>
```

### simulateWater.py

This script runs a short simulation of a water box and reports the speed and memory use.  Run it was

```
python simulateSpice.py <width> <model>
```

where `<width>` is the width of the water box in nm.  PDB files for water boxes of width 2, 3, 4, 5, and 6 are found
in the `water` directory.

### simulateSolvated.py

This script runs a simulation of a solvated molecule and checks for problems in simulation stability.  Run it as

```
python simulateSolvated.py <model>
```

### models.py

This file is used by the other scripts listed above.  It defines the supported MLIPs, including code to create an ASE
calculator for each one and to set the total charge and spin multiplicity.