/**
 * @file
 * Typically, input/output files will contain SI values that have to be converted
 * to/from internal units.
 *
 * The units struct provides an easy way to keep track of the internal simulation units.
 * A new unit system is created with new_units(), after which aexch and msat can be set.
 * Then, unitlength, unittime, unitfield and unitenergy can be called to get the respective
 * internal units expressed in SI units.
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


#ifdef __cplusplus
}
#endif
#endif