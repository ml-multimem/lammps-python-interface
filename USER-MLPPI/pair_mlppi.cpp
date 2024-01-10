/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_mlppi.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "library.h"        /* this is a LAMMPS include file */

#include <random>

#include <bits/stdc++.h> /* for tokenization */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMlppi::PairMlppi(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairMlppi::~PairMlppi()
{
  if (allocated) {
    printf("Cleaning up...\n");
    memory->destroy(setflag);

    memory->destroy(cut);
    memory->destroy(cutsq);
    printf("Cleaning up...DONE.\n");
  }
}

/* ---------------------------------------------------------------------- */

void PairMlppi::compute(int eflag, int vflag)
{
  // If I have not been initialized
  printf("The value of initialize is %d \n", this->initialized); 
  if (!this->initialized) {
    printf("Entering initialization\n");
    // Build the init string    
    const char *model_name = (const char *) lammps_extract_variable(lmp,"mlppi_model_name",NULL);
    printf("The value of variable %s is %s\n", "mlppi_model_name", model_name);
  
    const char *output_from_init_model = (const char *) lammps_extract_variable(lmp,"mlppi_initialize",NULL);

    // Here we have been initialized
    this->initialized = true;
    printf("End of init: the value of initialize is %d \n", this->initialized);
  }
    
  double **forces = atom->f;
  int n_particles = list->inum;
  
  // We create a serialized version of the info regarding our 
  // configuration that are required by the model (e.g. positions 
  // and atom types) and store it within the input variable 
  // mlppi_input which must be defined in the .in file.
  // The format is documented under create_input_for_python
  char *input_for_python_model = create_input_for_python(); 
  lammps_set_variable(lmp, "mlppi_input", input_for_python_model); 
  free(input_for_python_model); // Release 

  // Request answer from the external python model.
  // We evaluate the mlppi_forces python variable, which must be 
  // defined in the .in script. Evaluating mlppi_forces triggers
  // the evaluation of the corresponding python function defined 
  // in the .in script, using as input the value stored in mlppi_input
  // The format is documented under parse_output_from_python
  const char *output_from_python_model = (const char *) lammps_extract_variable(lmp,"mlppi_forces",NULL);
  // DEBUG LINE
  printf("The value of variable %s is %s\n", "mlppi_forces", output_from_python_model);  
  
  int n_rows, n_cols;  // Updated inside parse_output_from_python and used for memory deallocation
  double **parsed_output = this->parse_output_from_python(output_from_python_model, n_rows, n_cols);
  // lammps_free((void *)output_from_python_model); // Do NOT release because in the docs it says "do not free char* variables"
  // (https://docs.lammps.org/Library_objects.html#_CPPv423lammps_extract_variablePvPKcPKc)
  // "For other variable styles the returned pointer needs to be cast to a char pointer. It should not be deallocated."

  // Store parsed_output in the forces array
  for (int i_particle = 0; i_particle < n_particles; i_particle++) {
      forces[i_particle][0] = parsed_output[0][0];
      forces[i_particle][1] = parsed_output[0][1];
      forces[i_particle][2] = parsed_output[0][2];
  }

  // Clean up memory
  free_parsed_variable_memory(parsed_output, n_rows, n_cols);

}

void PairMlppi::free_parsed_variable_memory(double** variable_to_free, int n_rows, int n_cols) {
  // For each row
  for (int current_row = 0; current_row < n_rows; current_row++) {
      free(variable_to_free[current_row]); // Free memory for columns
  }

  // Then free memory for rows
  free(variable_to_free);

}

double** PairMlppi::parse_output_from_python(const char *output_from_python_model, int& n_rows, int& n_cols) {
  int row_count = 0;
  int col_count = 0;
  
  // Read in the rows and columns
  sscanf(output_from_python_model, "%d %d", &row_count, &col_count);
  // Update output variables
  n_rows = row_count;
  n_cols = col_count;

  // Move twice 
  output_from_python_model = move_to_next_token(move_to_next_token(output_from_python_model, ' '), '\t');
  
  // Allocate memory
  double **parsed_output = (double **)malloc(row_count * sizeof(double *));
  // For each row
  for (int current_row = 0; current_row < row_count; current_row++) {
    // Allocate memory
    parsed_output[current_row] = (double *)malloc(col_count * sizeof(double));
    // For each column
    for (int current_col = 0; current_col < col_count; current_col++) {
      // Examine whether we have more text to consume
      if (output_from_python_model == NULL) {
        // If not, something was wrong
        error->all(FLERR, "Force field information string was shorter than declared. Examine the maximum allowed length of the variable in a python command, if applicable...");
      }
      double current_value = 0;
      // Read value
      sscanf(output_from_python_model, "\t%lf", &current_value);
      // Update element
      parsed_output[current_row][current_col] = current_value;
      // Move to next token
      output_from_python_model = move_to_next_token(output_from_python_model, '\t');
    }
  }

  return parsed_output;
}


char* PairMlppi::create_input_for_python() {
  // The python model will receive a tab-separated list containing:
  // - the number of particles
  // - for each particle:
  //    - the particle type
  //    - x, y, z position
  
  // Read input params
  double **positions = atom->x;
  int *particle_types = atom->type;
  int n_particles = list->inum;
  // TODO: Examine if we need to add box info, image flags and whether the atom->x information is already wrapped

  // Allocate memory
  // Number of particles: 10 digits float (including point), plus separator tab (1 character)
  const int DECIMAL_PRECISION = 10;
  const int MAX_FLOAT_LENGTH = 15;
  const int NUM_DIMENSIONS = 3;
  const int DIGITS_FOR_TYPE = 6;
  //const int DIGITS_FOR_ID = 10; // Not used yet

  int allocateSize = (MAX_FLOAT_LENGTH+1) * sizeof(char) +  // 1 additional position for tabs
    list->inum * ( // Per particle
      // Coordinates per particle will take up: MAX_FLOAT_LENGTH digits float (including point) x NUM_DIMENSIONS coordinates, plus separator tab (1 character)
      (MAX_FLOAT_LENGTH + 1) * NUM_DIMENSIONS * sizeof(char) + // 1 for tab
      // Type per particle will take up: DIGITS_FOR_TYPE digits integer, plus separator tab (1 character)
      (DIGITS_FOR_TYPE  + 1) * sizeof(char) // 1 for tab
    )
    + 1; // The last zero

  char* input_string = (char *)malloc(allocateSize);
  // TODO: Add box and related info as needed (see above)

  int offset = 0;
  // TODO: Check boundaries concerning MAX_FLOAT_LENGTH
  
  // Prepare formats for output based on constants
  char typeFormat[100];
  sprintf(typeFormat, "%%%dd\t", DIGITS_FOR_TYPE);
  char floatFormat[100];
  sprintf(floatFormat, "%%%d.%dlf\t", MAX_FLOAT_LENGTH, DECIMAL_PRECISION); // Using lf for double precision.

  // Output num of particles
  offset += sprintf(input_string, floatFormat, (float)n_particles);

  // For each particle
  for (int i_particle = 0; i_particle < n_particles; i_particle++) {
    // TODO: Check boundaries concerning length constants
    // Type
    offset += sprintf(input_string + offset, typeFormat, particle_types[i_particle]);
    // Print info
    // Coordinates
    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
      offset += sprintf(input_string + offset, floatFormat, positions[i_particle][dim]);
    }

  }

  // DEBUG LINES
  ////std::cout << "Serialized input to python code: " << input_string << std::endl;;
  //////////////

  // Return result pointer
  return input_string;
}

const char* PairMlppi::move_to_next_token(const char *serialized_data, char separator) {
  int end_of_data = strlen(serialized_data);
  int cursor_position = 0;
  // While we have not reached the end of the string (minus 1)
  while ((cursor_position < end_of_data - 1) && (*(serialized_data + cursor_position) != separator)) 
    // Move to next position
    cursor_position++;
  
  cursor_position++; // Move beyond the separator
  if (cursor_position == end_of_data) // If we reached the end
    return NULL; // return Not Found
  
  return serialized_data + cursor_position; // Else return position found
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMlppi::allocate()
{
  printf("Allocating...\n");
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 1; // TODO: Revisit why we need 1 vs 0 here
      

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  printf("Allocating...DONE.\n");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMlppi::settings(int narg, char **arg)
{
  printf("Settings...\n");

  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
  printf("Settings...DONE.\n");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMlppi::coeff(int narg, char **arg)
{
    printf("Number of coefficients provided: %d... Actually ignoring all... :)\n", narg);
    if (!allocated) allocate();

    int count = 0;
    for (int i = 1; i <= atom->ntypes; i++) {
      for (int j = 1; j <= atom->ntypes; j++) {
        cut[i][j] = cut_global;
        setflag[i][j] = 1;
        count++;
      }
    }

    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
//     if (narg > 2) 
//      error->all(FLERR,"Incorrect args for pair coefficients: 2 coefficients needed per atom type for MLPPI.");
    
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMlppi::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"MLPPI (init_one) All pair coeffs are not set");

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMlppi::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMlppi::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMlppi::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMlppi::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMlppi::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,atom->f[i][0],atom->f[i][1],atom->f[i][2]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMlppi::write_data_all(FILE *fp)
{
    // TODO: We do not write about pairs
}

/* ---------------------------------------------------------------------- */

double PairMlppi::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj,
                         double &fforce)
{
    error->all(FLERR,"MLPPI single() pair function not implemented yet...");
}

/* ---------------------------------------------------------------------- */

void *PairMlppi::extract(const char *str, int &dim)
{
  dim = 0;

  error->all(FLERR,"MLPPI extract pair function not implemented yet...");
  return nullptr;
}
