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
// 
// double plength(param* u){
//   return sqrt(2. * u->aexch / (u->mu0 * u->msat*u->msat) );
// }
// 
// double ptime(param* u){
//   return 1.0 / (u->gamma0 * u->msat);
// }
// 
// double pfield(param* u){
//   return u->mu0 * u->msat;
// }
// 
// double penergy(param* u){
//   return u->aexch * plength(u);
// }
// 
// void printparam(param* u){
//   printf("msat      :\t%g A/m\n",   u->msat);
//   printf("aexch     :\t%g J/m\n",   u->aexch);
//   printf("mu0       :\t%g N/A^2\n", u->mu0);
//   printf("gamma0    :\t%g m/As\n",  u->gamma0);
//   printf("plength:\t%g m\n",     plength(u));
//   printf("ptime  :\t%g s\n",     ptime(u));
//   printf("penergy:\t%g J\n",     penergy(u));
//   printf("pfield :\t%g T\n",     pfield(u));
// }

//   fmt.Fprintln(out, "Material parameters");
//   fmt.Fprintln(out, "AExch      : \t", param.AExch, " J/m");
//   fmt.Fprintln(out, "MSat       : \t", param.MSat, " A/m");
//   fmt.Fprintln(out, "Gamma0      : \t", param.Gamma0, " m/As");
//   fmt.Fprintln(out, "Mu0     : \t", param.Mu0, " N/A^2");
//   fmt.Fprintln(out, "exch length: \t", param.UnitLength(), " m");
//   fmt.Fprintln(out, "p time  : \t", param.UnitTime(), " s");
//   fmt.Fprintln(out, "p energy: \t", param.UnitEnergy(), " J");
//   fmt.Fprintln(out, "p field: \t", param.UnitField(), " T");
//   fmt.Fprintln(out, "Geometry");
//   fmt.Fprintln(out, "Grid Size  : \t", param.Size);
//   fmt.Fprint(out, "Cell Size  : \t");
//   for i:=range(param.CellSize){
//     fmt.Fprint(out, param.UnitLength() * param.CellSize[i], " ");
//   }
//   fmt.Fprint(out, "(m), (");
//    for i:=range(param.CellSize){
//     fmt.Fprint(out, param.CellSize[i], " ");
//   }
//   fmt.Fprintln(out, "exch. lengths)");
//
//   fmt.Fprint(out, "Sim Size   : \t ");
//   for i:=range(param.Size){
//     fmt.Fprint(out, float(param.Size[i]) * param.UnitLength() * param.CellSize[i], " ");
//   }
//   fmt.Fprintln(out, "(m)");

#ifdef __cplusplus
}
#endif