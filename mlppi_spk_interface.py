import torch
import os
import schnetpack as spk
import schnetpack.transform as trn
import numpy as np
from ase import Atoms
import numpy as np
from platform import python_version

# REQUIREMENTS 
# - Install python headers (apt install python-dev [or python3-dev]) on the machine
# - make yes-PYTHON
# - install all python dependencies in the related environment
# - run lammps activating the related python environment

print("** PYTHON VERSION: " , python_version())

force_field = None
bInitialized = False
ml_model = None

def initialize_model(model_name: str = None): 

    # Initialize what is needed in the force field calculation
    # if it has not already been done
    global bInitialized

    if not bInitialized:
        bInitialized = True

    print ("** PYTHON: Loading model: ", model_name)

    import os.path

    savedir = os.getcwd() + "/" #os.path.dirname(__file__) # Use current dir
    global ml_model
    ml_model = torch.load(savedir + model_name)

    return "finished_loading_model"

def deserializeParticleData(particleDataString: str) -> tuple:
    # TODO: Examine parallelization effect

    # Expected format
    # The expected format is as follows:
        # Read every step
        # Positions (list of floats, with a size as a function of particles, every step)
        # Box information (list, with a fixed size, every step)

        # Utility parameters
        # Logging prepending

        # Future additions:
        # Temperature (float, every step)
        # Pressure (float, every step)
        # Custom init parameters (string, once)
        # Custom parameters (string, everytime)

    # [NumOfParticles] [particle positions as a series of numbers]

    fields = particleDataString.split()
    iNumOfParticles = int(float(fields[0]))
    particlePositions = [float(x) for x in fields[1:]]
    
    return reshapeParticleData(iNumOfParticles, particlePositions)

def reshapeParticleData(iNumOfParticles: int, particlePositions: list) -> np.array:

    temp_array = np.array(particlePositions).reshape(iNumOfParticles, 4)
    positions_3d = temp_array[:, 1:4]
    particleTypes = temp_array[:, 0]
    # DEBUG LINES
    print(f"++ PYTHON: {str(positions_3d)} ")
    
    return particleTypes, positions_3d

def compute_forces(particleDataString = None):

    if particleDataString is not None:
        # DEBUG LINES
        print("++ PYTHON: Received particle data! START")
        print(particleDataString)
        print("-- PYTHON: Received particle data! END")
        print("++ Parsing and outputting particle positions.")
        
        particleTypes, particlePositions = deserializeParticleData(particleDataString)
        
        # DEBUG LINES
        print("-- PYTHON: Parsing and outputting particle positions. END")
        print(particlePositions)
    else:
        print("** PYTHON: Nothing received... :(")
        particlePositions = []
    
    if len(particlePositions) == 0:
        print("** PYTHON: Nothing requested in essence. Returning nothing courteously. :)")
        return serialize(np.array([[]]))

    # Generate appropriate data for the model
    # - _idx: number of instance/frame
    # - _n_atoms : torch.Size([1])
    # - _atomic_numbers : torch.Size([n_particles])
    # - _positions : torch.Size([n_particles, 3])
    # - _cell : torch.Size([1, 3, 3])
    # - _pbc : torch.Size([3])    

    # Cell info. TODO: Get it from the simulator
    print("++ PYTHON: Creating model input...")
    box_length = np.array([5.973166,5.973166,5.973166])

    onecell = np.array([[box_length[0], 0.0000,         0.0000],
                        [ 0.0000,       box_length[1],  0.0000],
                        [ 0.0000,       0.0000,         box_length[2]]]) 

    # set up converter
    # TODO: Get cutoff from the simulator 
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32
    )

    atoms = Atoms(
        numbers=np.array([1 for _ in range(len(particlePositions))]), 
        positions=particlePositions,
        cell = onecell,
        pbc = True
    )
    
    # DEBUG LINES
    print(str(atoms))
    print(str(atoms.numbers))
    print(str(atoms.positions))
    info_that_the_model_needs = converter(atoms) # TODO: Update
    print("-- PYTHON: Creating model input... DONE.")

    # Get the model output
    print("++ PYTHON: Calling the model...")
    force_field = ml_model(info_that_the_model_needs)
    print("-- PYTHON: Calling the model... DONE.")

    # TODO: Return perhaps everything
    return serialize(force_field['forces'])

def serialize(numpyArray):
    iRows, iCols = numpyArray.shape

    sResultString = "%d %d"%(iRows, iCols)
    for fComponent in numpyArray.flatten():
        sResultString += "\t%10.8f"%(fComponent)
    
    return sResultString.strip()
