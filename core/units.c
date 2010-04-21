#ifdef __cplusplus
extern "C" {
#endif

#include "units.h"
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
  return sqrt(2. * u->aexch / u->mu0 * u->msat*u->msat);
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


#ifdef __cplusplus
}
#endif