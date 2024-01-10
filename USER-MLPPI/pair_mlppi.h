/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov 

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlppi,PairMlppi)

#else

#ifndef LMP_PAIR_MLPPI_H
#define LMP_PAIR_MLPPI_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMlppi : public Pair {
 public:
  PairMlppi(class LAMMPS *);
  virtual ~PairMlppi();
  virtual void compute(int, int);

  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  bool initialized = false;
  double cut_global;
  double **cut;

  char* create_input_for_python();
  double** parse_output_from_python(const  char *output_from_python_model, int& n_rows, int& n_cols);

  void free_parsed_variable_memory(double** variable_to_free, int n_rows, int n_cols);

  virtual void allocate();

 private:
  const char* move_to_next_token(const char *serialized_data, char separator);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
