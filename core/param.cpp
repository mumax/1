#ifdef __cplusplus
extern "C" {
#endif

#include "param.h"


param* new_param(){
  param* p = (param*)malloc(sizeof(param));
  
  p->aexch = 0.;
  p->msat = 0.;
  p->mu0 = 4.0E-7 * PI;
  p->gamma0 = 2.211E5;
  p->alpha = 0.;

  p->anisType = NONE;
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

  p->solverType = NONE;
  p->maxDt = 0.;
  p->maxDelta = 0.;
  p->maxError = 0.;

  p->exchType = NONE;
  
  return p;
}

void check_param(param *p){

  // there must be a valid unit system
  assert(p->msat > 0.);
  assert(p->aexch > 0.);
  assert(p->mu0 > 0.);
  assert(p->gamma0 > 0.);
  
  // no negative damping
  assert(p->alpha >= 0.);
  
  // thickness 1 only allowed in x-direction
  assert(p->size[X]>0);
  assert(p->size[Y]>1);
  assert(p->size[Z]>1);
  
  for (int i=0; i<3; i++){
    assert( p->cellSize[i]>0.0f);
    assert( p->demagCoarse[i]>0);
    // the coarse level mesh should fit the low level mesh:
    assert( p->size[i]>p->demagCoarse[i] && p->size[i]%p->demagCoarse[i] == 0);
  }

  // only 1 (possibly coarse level) cell thickness in x-direction combined with periodicity in this direction is not allowed.
  assert(  !(p->size[X]/p->demagCoarse[X]==1 && p->demagPeriodic[X])  );     
  return;
}

double unitlength(param* p){
  return sqrt(2. * p->aexch / (p->mu0 * p->msat*p->msat) );
}

double unittime(param* p){
  return 1.0 / (p->gamma0 * p->msat);
}

double unitfield(param* p){
  return p->mu0 * p->msat;
}

double unitenergy(param* p){
  return p->aexch * unitlength(p);
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

  fprintf(out, "kernelType   :\t%d\n",       p->kernelType);
  fprintf(out, "demagPeriodic:\t[%d, %d, %d] repeats\n", p->demagPeriodic  [X], p->demagPeriodic[  Y], p->demagPeriodic  [Z]);
  fprintf(out, "demagCoarse  :\t[%d x %d x %d] cells\n", p->demagCoarse    [X], p->demagCoarse    [Y], p->demagCoarse    [Z]);
  fprintf(out, "demagKernel  :\t[%d x %d x %d] cells\n", p->kernelSize[X], p->kernelSize[Y], p->kernelSize[Z]);

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