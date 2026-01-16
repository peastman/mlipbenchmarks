This directory contains files related to the SPICE test set.  `SPICE-test.hdf5` contains the dataset, and was downloaded
from https://zenodo.org/records/17620280.  The CSV files were created with the `computeSpice.py` script in the parent
directory.  There is one for each model.  For each molecule or dimer in the dataset they list the number of atoms, the
total charge, and the mean absolute error (MAE) in energy differences between conformations.  MAE values in these files
are in kJ/mol, unlike the paper which reports them in kcal/mol.

The `createSolvatedMolecule.py` script was used to create the `solvated.pdb` file used to test simulation stability.