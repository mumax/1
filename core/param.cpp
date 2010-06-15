#ifdef __cplusplus
extern "C" {
#endif

#include "param.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


param* new_param(){
  param* p = (param*)malloc(sizeof(param));
  
  p->aexch = 0.;
  p->msat = 0.;
  p->mu0 = 4.0E-7 * PI;
  p->gamma0 = 2.211E5;
  p->alpha = 0.;

  p->anisType = ANIS_NONE;
  p->anisK = NULL;
  p->anisN = 0;

  for(int i=0; i<3; i++){
    p->size[i] = 0;
    p->cellSize[i] = 0.;
    p->demagPeriodic[i] = 0;
    p->demagCoarse[i] = 1;    
  }

  p->solverType = SOLVER_NONE;
  p->maxDt = 0.;
  p->maxDelta = 0.;
  p->maxError = 0.;
  
  return p;
}

double unitlength(param* u){
  return sqrt(2. * u->aexch / (u->mu0 * u->msat*u->msat) );
}

double unittime(param* u){
  return 1.0 / (u->gamma0 * u->msat);
}

double unitfield(param* u){
  return u->mu0 * u->msat;
}

double unitenergy(param* u){
  return u->aexch * unitlength(u);
}

void printparam(FILE* out, param* u){
  fprintf(out, "msat      :\t%g A/m\n",   u->msat);
  fprintf(out, "aexch     :\t%g J/m\n",   u->aexch);
  fprintf(out, "mu0       :\t%g N/A^2\n", u->mu0);
  fprintf(out, "gamma0    :\t%g m/As\n",  u->gamma0);
  fprintf(out, "unitlength:\t%g m\n",     unitlength(u));
  fprintf(out, "unittime  :\t%g s\n",     unittime(u));
  fprintf(out, "unitenergy:\t%g J\n",     unitenergy(u));
  fprintf(out, "unitfield :\t%g T\n",     unitfield(u));
}



#ifdef __cplusplus
}
#endif