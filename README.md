# lammps-python-interface

LAMMPS pair style that allows to interface with Python, in order to use ML potentials for forces calculations.
The interface is agnostic with respect to the specific Python framework utilized to create the ML model: the model loading, initialization, and data formatting is handled by separate python functions, contained in a file specified within the LAMMPS input script.

The interface makes use of: (1) the `pair_style mlppi` (machine-learned potential python interface) (2) the `python` command and (3) python `variables`. 

The python command allows interfacing LAMMPS with an embedded Python interpreter and registering a Python function for future execution. The Python function is assigned to a python-style variable definedin the input script. Whenever the variable is evaluated, it will execute the Python function to assign a value to the variable.

The names that refer to C++ variables within "pair_mlppi.cpp" must be kept as follows:
- `mlppi_forces` 
- `mlppi_input`

The names referring to python functions can be modified, as long as they are the same in the lammps .in and in the .py script called by the lammps python variables

## Python environment configuration
The supplied sample model was generated using the SchNetPack package [1], which require the following environment configuration. Adapt this step to the requirements of your ML framework.
```conda create --name schnetpack python=3.8
conda activate schnetpack
git clone git@github.com:ml-multimem/schnetpack.git
cd schnetpack
pip3 install --user -r requirements.txt
pip3 install --user .
```

## LAMMPS build customizations
Tested with LAMMPS version: lammps-23Jun2022.

In order to allow LAMMPS to locate the python interpreter, you should edit **Makefile.lammps** with the location of the python executable, after the configuration of the environment. Location of the Makefile.lammps file --> /lib/python/

Then build LAMMPS adding the USER-MLPPI package and other relevant packages:
```cd /path/to/lammps/lammps-23Jun2022/src/
make clean-all
make yes-USER-MLPPI
make yes-CLASS2
make yes-KSPACE
make yes-MOLECULE
make yes-PYTHON
make serial -j
```

## Test with the sample input and model

`./path/to/lammps/executable/lmp_serial -in md_with_mlppi.in > log.lammps`

[1] K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger, *SchNetPack 2.0: A neural network toolbox for atomistic machine learning*, J. Chem. Phys. 158 (2023) 144801. https://doi.org/10.1063/5.0138367.
