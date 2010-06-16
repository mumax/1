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
    p->hExt[i] = 0;
    p->diffHExt[i] = 0;
  }

  p->solverType = SOLVER_NONE;
  p->maxDt = 0.;
  p->maxDelta = 0.;
  p->maxError = 0.;

  p->exchType = EXCH_NONE;
  
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

void param_print(FILE* out, param* p){
  fprintf(out, "msat      :\t%g A/m\n",   p->msat);
  fprintf(out, "aexch     :\t%g J/m\n",   p->aexch);
  fprintf(out, "mu0       :\t%g N/A^2\n", p->mu0);
  fprintf(out, "gamma0    :\t%g m/As\n",  p->gamma0);
  fprintf(out, "alpha     :\t%g\n",       p->alpha);

  fprintf(out, "anisType  :\t%d\n",       p->anisType);
  fprintf(out, "anisK     :\t[");
  for(int i=0; i<p->anisN; i++){
    fprintf(out, "%g ", p->anisK[i]);
  }
  fprintf(out, "]\n");
  
  fprintf(out, "unitlength:\t%g m\n",     unitlength(p));
  fprintf(out, "unittime  :\t%g s\n",     unittime(p));
  fprintf(out, "unitenergy:\t%g J\n",     unitenergy(p));
  fprintf(out, "unitfield :\t%g T\n",     unitfield(p));

  
}



#ifdef __cplusplus
}
#endif