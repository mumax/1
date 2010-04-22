#ifdef __cplusplus
extern "C" {
#endif

#include "units.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


units* new_units(){
  units* unit = (units*)malloc(sizeof(units));
  unit->aexch = 0.;
  unit->msat = 0.;
  unit->mu0 = 4.0E-7 * PI;
  unit->gamma0 = 2.211E5;
  return unit;
}

double unitlength(units* u){
  return sqrt(2. * u->aexch / (u->mu0 * u->msat*u->msat) );
}

double unittime(units* u){
  return 1.0 / (u->gamma0 * u->msat);
}

double unitfield(units* u){
  return u->mu0 * u->msat;
}

double unitenergy(units* u){
  return u->aexch * unitlength(u);
}

void printunits(units* u){
  printf("msat      :\t%g A/m\n",   u->msat);
  printf("aexch     :\t%g J/m\n",   u->aexch);
  printf("mu0       :\t%g N/A^2\n", u->mu0);
  printf("gamma0    :\t%g m/As\n",  u->gamma0);
  printf("unitlength:\t%g m\n",     unitlength(u));
  printf("unittime  :\t%g s\n",     unittime(u));
  printf("unitenergy:\t%g J\n",     unitenergy(u));
  printf("unitfield :\t%g T\n",     unitfield(u));
}

//   fmt.Fprintln(out, "Material parameters");
//   fmt.Fprintln(out, "AExch      : \t", units.AExch, " J/m");
//   fmt.Fprintln(out, "MSat       : \t", units.MSat, " A/m");
//   fmt.Fprintln(out, "Gamma0      : \t", units.Gamma0, " m/As");
//   fmt.Fprintln(out, "Mu0     : \t", units.Mu0, " N/A^2");
//   fmt.Fprintln(out, "exch length: \t", units.UnitLength(), " m");
//   fmt.Fprintln(out, "unit time  : \t", units.UnitTime(), " s");
//   fmt.Fprintln(out, "unit energy: \t", units.UnitEnergy(), " J");
//   fmt.Fprintln(out, "unit field: \t", units.UnitField(), " T");
//   fmt.Fprintln(out, "Geometry");
//   fmt.Fprintln(out, "Grid Size  : \t", units.Size);
//   fmt.Fprint(out, "Cell Size  : \t");
//   for i:=range(units.CellSize){
//     fmt.Fprint(out, units.UnitLength() * units.CellSize[i], " ");
//   }
//   fmt.Fprint(out, "(m), (");
//    for i:=range(units.CellSize){
//     fmt.Fprint(out, units.CellSize[i], " ");
//   }
//   fmt.Fprintln(out, "exch. lengths)");
//   
//   fmt.Fprint(out, "Sim Size   : \t ");
//   for i:=range(units.Size){
//     fmt.Fprint(out, float(units.Size[i]) * units.UnitLength() * units.CellSize[i], " ");
//   }
//   fmt.Fprintln(out, "(m)");

#ifdef __cplusplus
}
#endif