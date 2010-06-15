/**
 * @file
 *
 * @deprecated Superseeded by param.h
 *
 * The simulations internally use program units, which are adapted to the intrinsic scale of the problem. 
 * This avoids having to deal with extremely large or small quantities, which may lead to numerical errors. 
 * Our program units are chosen so that the values of the following four quantities become one:
 * 
 *	- msat (saturation magnetization of the used material, A/m)
 *	- eaxch (exchange constant of the used material, J/m)
 *	- gamma0 (gyromagnetic ratio, 2.211E5 m/As)
 *	- mu0 (permeability of vacuum, 4pi*1E-7 N/A^2)
 *
 * This leads to the following set of program units:
 *
 *	- Unit of field: msat * mu0
 *	- Unit of length: sqrt(2aexch/(mu0 * msat^2)), the exchange length
 *	- Unit of time:1/(gamma0 * msat)
 * 	- Unit of energy: aexch * unitlength
 *
 * The units struct provides an easy way to keep track of the internal simulation units.
 * A new unit system is created with new_units(), after which aexch and msat can be set.
 * Then, unitlength(), unittime(), unitfield() and unitenergy() can be called to get the respective
 * internal units expressed in SI units.
 * 
 * @note gamma0 and mu0 are set to their default values, but can be changed if desired.
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef UNITS_H
#define UNITS_H

#define PI 3.14159265

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  double aexch;		///< Exchange constant in J/m
  double msat;		///< Saturation magnetization in A/m
  double mu0;		///< Mu0 in N/A^2
  double gamma0;	///< Gyromagnetic ratio in m/As
  
}units;

/** New units with zero msat and aexch, default values for mu0 and gamma0. */
units* new_units();

/** The internal unit of length, expressed in meters. */
double unitlength(units* u);

/** The internal unit of time, expressed in seconds. */
double unittime(units* u);

/** The internal unit of field, expressed in tesla. */
double unitfield(units* u);

/** The internal unit of energy, expressed in J. */
double unitenergy(units* u);

/** Prints the material parameters and units to stderr */
void printunits(units* u);

#ifdef __cplusplus
}
#endif
#endif