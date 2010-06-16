#ifdef __cplusplus
extern "C" {
#endif

#include "param.h"
#include "tensor.h"
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
  
  fprintf(out, "msat         :\t%g A/m\n",   p->msat);
  fprintf(out, "aexch        :\t%g J/m\n",   p->aexch);
  fprintf(out, "mu0          :\t%g N/A^2\n", p->mu0);
  fprintf(out, "gamma0       :\t%g m/As\n",  p->gamma0);
  fprintf(out, "alpha        :\t%g\n",       p->alpha);

  fprintf(out, "anisType     :\t%d\n",       p->anisType);
  fprintf(out, "anisK        :\t[");
  for(int i=0; i<p->anisN; i++){
    fprintf(out, "%g ", p->anisK[i]);
  }
  fprintf(out, "]\n");

  double L = unitlength(p);
  fprintf(out, "size         :\t[%d x %d x %d] cells\n", p->size[X], p->size[Y], p->size[Z]);
  fprintf(out, "cellsize     :\t[%g m x %g m x %g m]\n", p->cellSize[X]*L, p->cellSize[Y]*L, p->cellSize[Z]*L);

  fprintf(out, "demagPeriodic:\t[%d, %d, %d] repeats\n", p->demagPeriodic  [X], p->demagPeriodic[  Y], p->demagPeriodic  [Z]);
  fprintf(out, "demagCoarse  :\t[%d x %d x %d] cells\n", p->demagCoarse    [X], p->demagCoarse    [Y], p->demagCoarse    [Z]);
  fprintf(out, "demagKernel  :\t[%d x %d x %d] cells\n", p->demagKernelSize[X], p->demagKernelSize[Y], p->demagKernelSize[Z]);

  fprintf(out, "exchType     :\t%d\n",       p->exchType);

  double T = unittime(p);
  fprintf(out, "solverType   :\t%d\n",       p->solverType);
  fprintf(out, "maxDt        :\t%g s\n",     p->maxDt * T);
  fprintf(out, "maxDelta     :\t%g\n",       p->maxDelta);
  fprintf(out, "maxError     :\t%g\n",       p->maxError);

  double B = unitfield(p);
  fprintf(out, "hExt         :\t[%g, %g, %g] T\n", p->hExt[X]*B, p->hExt[Y]*B, p->hExt[Z]*B);
  fprintf(out, "diffHExt     :\t[%g, %g, %g] T/s\n", p->diffHExt[X]*(B/T), p->diffHExt[Y]*(B/T), p->diffHExt[Z]*(B/T));
  
  fprintf(out, "unitlength   :\t%g m\n",     unitlength(p));
  fprintf(out, "unittime     :\t%g s\n",     unittime(p));
  fprintf(out, "unitenergy   :\t%g J\n",     unitenergy(p));
  fprintf(out, "unitfield    :\t%g T\n",     unitfield(p));
  
}



#ifdef __cplusplus
}
#endif