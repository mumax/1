/**
 * @file
 *
 * The params struct stores simulation parameters like
 *  - material constants
 *  - simulation size
 *  - demag type
 *  - time stepping parameters
 * By convention, these parameters are only changed by the user,
 * @em not by the simulation itself (e.g.: a solver itself must not
 * change the maxDt parameter, etc.).
 *
 * Should there be a need for a central place to store all variables
 * that @em are changed during the simulation (like m, h, time, ...),
 * then a second struct can be made for this.
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
 * @author Arne Vansteenkiste, Ben Van de Wiele
 *
 */
#ifndef PARAM_H
#define PARAM_H

#include "assert.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#define PI 3.14159265

#ifdef __cplusplus
extern "C" {
#endif


/**
 * 
 */
typedef struct{

  //Material parameters
 
  double aexch;         ///< Exchange constant in J/m
  double msat;          ///< Saturation magnetization in A/m
  double mu0;           ///< Mu0 in N/A^2
  double gamma0;        ///< Gyromagnetic ratio in m/As
  double alpha;         ///< Damping constant (dimensionless)

  int anisType;         ///< Identifies the anisotropy type (see typedefs ANIS_NONE, ANIS_UNIAXIAL, ...)
  float* anisK;         ///< Anisotropy constants, as many as needed by the anisotropy type
  int anisN;            ///< Number of ansiotropy constants stored in anisK


  // FD geometry

  int size[3];          ///< Number of cells in each direction. For 2D sims, the FIRST size should be 1
  float cellSize[3];    ///< Size of the FD cells (internal units)


  // demag settings

  int kernelType;
  
  int demagPeriodic[3];   ///< Periodic boundary conditions in each of the directions? 0 means no periodicity, a positive number means repeat N times in the positive and negative direction along the respective axis.
  int demagCoarse[3];     ///< Combine N cells in each respective direction into a larger cell, for a faster, coarser evaluation of the demag field. {1, 1, 1} means full resolution.
  int kernelSize[3];      ///< Size of the convolution kernel. In principle it can be derived from demagPeriodic and demagCoarse, but we store it anyway for convienience.
  ///@todo add demag kernel here too?


  // exchange settings;

  int exchType;
    /// Exchange types that can be added in the convolution should have a positive int assigned.
    /// Exchange types that can only be added classically should have a negative int assigned.
    /// exchType = 0 is not defined.
  int exchInConv[3];
    ///Depending on the kerneltype and/or a coarse grid evaluation of the demag field, some/all components of the exchange fields need to be added classically
    ///   exchInConv[comp] = -1 : for component 'comp' of the field, exchange is not computed (not classically, nor in convolution)
    ///   exchInConv[comp] =  0 : for component 'comp' of the field, exchange is computed classically 
    ///   exchInConv[comp] =  1 : for component 'comp' of the field, exchange is included in the convolution

  // more to come here

  
  // time stepping

  int solverType;       ///< Identifies the solver type (see typedefs SOLVER_EULER, SOLVER_HEUN, SOLVER_ANAL, ...)
  float maxDt;          ///< Time step (internal units). This applies only for solvers with a fixed time step, other solvers may ignore it or use it as an absolute maximum step.
  float maxDelta;       ///< The maximum "change" an adaptive step solver may make per step. Depending on the solver this may be, e.g., delta_m, delta_phi, ... Other solvers may ignore this.
  float maxError;       ///< The maximum error per step, for adaptive solvers. Others may ignore this.


  //normalization

  int normalizeEvery;   ///< Normalize the magnetization vectors after every N (partial) time steps
  tensor* msatMap;      ///< Space-dependent magenetization norm, typically contains numbers between 0 and 1. NULL means msat is uniform over space

  // field
  
  float hExt[3];        ///< The externally applied field (internal units)
  float diffHExt[3];    ///< Time derivative of the applied field (internal units). Most solvers ignore this, it should only be set for some special solver I have in mind.

}param;


/**
 * checks if the parameter list has acceptable values. 
 */
void check_param(param *p);

/// Possible value for any type (anisotropy, solver, exchange, ...)
#define NONE 0


// Anisotropy types

/// Possible value for anisType. Means uniaxial anisotropy.
#define ANIS_UNIAXIAL 1

/// Possible value for anisType. Means cubic anisotropy.
#define ANIS_CUBIC 2

/// Possible value for anisType. Means shape anisotropy.
#define ANIS_EDGE 3


// Solver types


/// Possible value for solverType. Simple Euler method
#define SOLVER_EULER 1

/// Possible value for solverType. 2nd order Heun method
#define SOLVER_HEUN 2

/// Possible value for solverType. 4th order Runge-Kutta
#define SOLVER_RK4 4

/// Possible value for solverType. 5th order Dormand-Prince with adaptive step size
#define SOLVER_DP45 45

/// Possible value for solverType. Ben Van de Wiele's forward semi-analytical solver
#define SOLVER_ANAL_FW 128

/// Possible value for solverType. Ben Van de Wiele's predictor/corrector semi-analytical solver
#define SOLVER_ANAL_PC 256

// Exchange types

/// Possible value for exchType. 6-neighbour exchange (3D geometry).
#define EXCH_6NGBR 6



// Kernel types

#define KERNEL_DIPOLE 1

/// Possible value for kernelType. Quantities only dependent on Y and Z coordinate, invariance in X direction. Includes exchange
#define KERNEL_MICROMAG2D 2

/// Possible value for kernelType. Classical 3D micromagnetic kernel. Can also be used for 2D simulations (1 cell in X). Includes exchange.
#define KERNEL_MICROMAG3D 3



// Methods

/// New param with default values.
param* new_param();

/// Frees everything. @todo implement
void delete_param(param* p);

/// The internal unit of length, expressed in meters.
double unitlength(param* p);

/// The internal unit of time, expressed in seconds.
double unittime(param* p);

/// The internal unit of field, expressed in tesla.
double unitfield(param* p);

/// The internal unit of energy, expressed in J.
double unitenergy(param* p);

/// Prints the parameters
void param_print(FILE* out, param* p);

#ifdef __cplusplus
}
#endif
#endif