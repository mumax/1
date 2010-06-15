/**
 * @file
 *
 * The params struct stores all the simulation parameters like
 *  - material constants
 *  - simulation size
 *  - demag type
 *  - time stepping parameters
 *
 * @warning Unless otherwise noted, all values represent internal units.
 * Conversion from SI units is typically done right after the input file is read,
 * and conversion back to SI is done right before output is written.
 * In between, everything is doen in internal units, even if not noted explicitly.
 *
 * The internal program units are well adapted to the intrinsic scale of the problem.
 * This avoids having to deal with extremely large or small quantities, which may lead to numerical errors.
 * Our program units are chosen so that the values of the following four quantities become one:
 *
 *  - msat (saturation magnetization of the used material, A/m)
 *  - eaxch (exchange constant of the used material, J/m)
 *  - gamma0 (gyromagnetic ratio, 2.211E5 m/As)
 *  - mu0 (permeability of vacuum, 4pi*1E-7 N/A^2)
 *
 * This leads to the following set of program units:
 *
 *  - Unit of field: msat * mu0
 *  - Unit of length: sqrt(2aexch/(mu0 * msat^2)), the exchange length
 *  - Unit of time:1/(gamma0 * msat)
 *  - Unit of energy: aexch * unitlength
 *
 * The param struct provides an easy way to keep track of the internal simulation units.
 * A new unit system is created with new_param(), after which aexch and msat can be set.
 * Then, unitlength(), unittime(), unitfield() and unitenergy() can be called to get the respective
 * internal units expressed in SI units.
 *
 * @note gamma0 and mu0 are set to their default values, but can be changed if desired.
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef PARAM_H
#define PARAM_H

#define PI 3.14159265

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Stores the material parameters
 */
typedef struct{

  double aexch;     ///< Exchange constant in J/m
  double msat;      ///< Saturation magnetization in A/m
  double mu0;       ///< Mu0 in N/A^2
  double gamma0;    ///< Gyromagnetic ratio in m/As
  double alpha;     ///< Damping constant (dimensionless)

  int anisType;     ///< Identifies the anisotropy type (see typedefs ANIS_NONE, ANIS_UNIAXIAL, ...)
  float* anisK;     ///< Anisotropy constants, as many as needed by the anisotropy type
  int anisN;        ///< Number of ansiotropy constants stored in anisK
    
}Material;


/// Possible value for material.anisType. Means no anisotropy.
#define ANIS_NONE 0

/// Possible value for material.anisType. Means uniaxial anisotropy.
#define ANIS_UNIAXIAL 1

/// Possible value for material.anisType. Means cubic anisotropy.
#define ANIS_CUBIC 2

/// Possible value for material.anisType. Means shape anisotropy.
#define ANIS_EDGE 3



/**
 *  Stores the FD geometry
 */
typedef struct{

  int size[3];          ///< Number of cells in each direction. For 2D sims, the FIRST size should be 1
  float cellSize[3];    ///< Size of the FD cells (internal units)
  
}Geometry;



/**
 * Stores the demag settings
 */
typedef struct{
  int periodic[3];    ///< Periodic boundary conditions in each of the directions? 0 means no periodicity, a positive number means repeat N times in the respective direction.
  int combine[3];     ///< Combine N cells in each respective direction into a larger cell, for a faster, coarser evaluation of the demag field. {1, 1, 1} means full resolution.
}Demag;



/**
 * Stores the time stepping settings
 */
typedef struct{
  int solverType;   ///< Identifies the solver type (see typedefs SOLVER_EULER, SOLVER_HEUN, SOLVER_ANAL, ...)
  double t;         ///< Total elapsed simulation time (internal units)
  double dt;        ///< Time step (internal units). This can be set manually for solvers with a fixed time step, while adaptive solvers should update this value to report the used time step.
}Time;


/// Possible value for time.solverType. Means no solver is set.
#define SOLVER_NONE 0

/// Possible value for time.solverType. Simple Euler method
#define SOLVER_EULER 1

/// Possible value for time.solverType. 2nd order Heun method
#define SOLVER_HEUN 2

/// Possible value for time.solverType. 4th order Runge-Kutta
#define SOLVER_RK4 4

/// Possible value for time.solverType. 5th order Dormand-Prince with adaptive step size
#define SOLVER_DP45 45

/// Possible value for time.solverType. Ben Van de Wiele's semi-analytical solver
#define SOLVER_ANAL 128


/**
 * Grand unification of the above
 */
typedef struct{
  
  Material material;
  Geometry geometry;
  Demag demag;
  Time time;
  
}Param;


/** New units with zero msat and aexch, default values for mu0 and gamma0. */
// units* new_units();

/** The internal unit of length, expressed in meters. */
// double unitlength(units* u);

/** The internal unit of time, expressed in seconds. */
// double unittime(units* u);

/** The internal unit of field, expressed in tesla. */
// double unitfield(units* u);

/** The internal unit of energy, expressed in J. */
// double unitenergy(units* u);

/** Prints the material parameters and units to stderr */
// void printunits(units* u);

#ifdef __cplusplus
}
#endif
#endif