# schnet-lammps-interface

Python env configuration
- conda create --name schnetpack python=3.8
- conda activate schnetpack
- git clone git@github.com:eleori/schnetpack-extension.git
- cd schnetpack-extension
- pip3 install --user -r requirements.txt
- pip3 install --user .

LAMMPS version: lammps-23Jun2022
- location of the "Makefile.serial" file --> /src/MAKE/
- location of the "Makefile.lammps" file --> /lib/python/

Makefile.lammps needs to be edited with the location of the python executable after the confiration of the environment 
