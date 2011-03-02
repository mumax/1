#include "cpu_exch.h"
#include "../macros.h"
#include <stdlib.h>
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_add_exch (float *m, float *h, int *size, int *periodic, int *exchInConv, float *cellSize, int type){

  if(exchInConv[X]!=0 && exchInConv[Y]!=0 && exchInConv[Z]!=0)
    return;
  
  float *m_comp = NULL;
  float *h_comp = NULL;

  int N = size[X]*size[Y]*size[Z];
  for (int i=0; i<3; i++)
    if (exchInConv[i]==0){
      m_comp = &m[i*N];
      h_comp = &h[i*N];
      
      switch (type){
        case EXCH_6NGBR:
          if (size[X] == 1)
            cpu_add_6NGBR_exchange_2D_geometry (m_comp, h_comp, size, periodic, cellSize);
          else 
            cpu_add_6NGBR_exchange_3D_geometry (m_comp, h_comp, size, periodic, cellSize);
          break;
        case EXCH_12NGBR:
          if (size[X] == 1)
            cpu_add_12NGBR_exchange_2D_geometry (m_comp, h_comp, size, periodic, cellSize);
          else 
            cpu_add_12NGBR_exchange_3D_geometry (m_comp, h_comp, size, periodic, cellSize);
          break;
        default:
          fprintf(stderr, "abort: no valid exchType %d\n", type);
          abort();
      }
    }

  return;
}



typedef struct{
  float *m, *h, *cellSize;
  float cst_y, cst_z;
  int *periodic;
  int Ny, Nz, Ntot;
} cpu_add_6NGBR_exchange_2D_geometry_arg;

void cpu_add_6NGBR_exchange_2D_geometry1_t(int id){

  cpu_add_6NGBR_exchange_2D_geometry_arg *arg = (cpu_add_6NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nz);
  for (pH = arg->h + start, pM = arg->m + arg->Nz + start; pH < arg->h + stop; pH++, pM++)                // add cst_y*M[y+1]
    *pH += arg->cst_y * (*pM);
  if (arg->periodic[Y] && id==0){
    for (pH = arg->h + arg->Ntot - arg->Nz, pM = arg->m; pH < arg->h + arg->Ntot; pH++, pM++)             // add cst_y*M[y=Ny] for periodic case
      *pH += arg->cst_y * (*pM);
  }

  return;
}

void cpu_add_6NGBR_exchange_2D_geometry2_t(int id){

  cpu_add_6NGBR_exchange_2D_geometry_arg *arg = (cpu_add_6NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nz);
  for (pH = arg->h + arg->Nz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                // add cst_y*M[y-1] 
    *pH += arg->cst_y * (*pM);
  if (arg->periodic[Y] && id==0){
    for (pH = arg->h, pM = arg->m + arg->Ntot - arg->Nz; pH < arg->h + arg->Nz; pH++, pM++)               // add cst_y*M[y=-1] for periodic case
      *pH += arg->cst_y * (*pM);
  }

  return;
}

void cpu_add_6NGBR_exchange_2D_geometry3_t(int id){

  cpu_add_6NGBR_exchange_2D_geometry_arg *arg = (cpu_add_6NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ny);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 1; pH<arg->h + (i+1)*arg->Nz-1; pH++, pM++)   // add cst_z*M[z+1]
      *pH += arg->cst_z * (*pM);
    if (arg->periodic[Z])                                                                                 // add cst_z*M[z=Nz] for periodic case
      arg->h[(i+1)*arg->Nz-1] += arg->cst_z*arg->m[i*arg->Nz];

    for (pH = arg->h + i*arg->Nz + 1, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)     // add cst_z*M[z-1]
      *pH += arg->cst_z * (*pM);
    if (arg->periodic[Z])                                                                                 // add cst_z*M[z=-1] for periodic case
      arg->h[i*arg->Nz] += arg->cst_z*arg->m[(i+1)*arg->Nz-1];
  }

  return;
}

void cpu_add_6NGBR_exchange_2D_geometry4_t(int id){

  cpu_add_6NGBR_exchange_2D_geometry_arg *arg = (cpu_add_6NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot);
  for (pH = arg->h + start, pM = arg->m + start; pH < arg->h + stop; pH++, pM++)                           // add (cst_y + cst_z) * M
    *pH -= 2.0f*(arg->cst_y + arg->cst_z) * (*pM);

  return;
}

void cpu_add_6NGBR_exchange_2D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize){
  
  cpu_add_6NGBR_exchange_2D_geometry_arg args;
  args.m = m;
  args.h = h;
  args.cellSize = cellSize;
  args.cst_y = 1.0f/cellSize[Y]/cellSize[Y];
  args.cst_z = 1.0f/cellSize[Z]/cellSize[Z];
  args.periodic = periodic;
  args.Ny = size[Y];
  args.Nz = size[Z];
  args.Ntot = size[Y]*size[Z];

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_6NGBR_exchange_2D_geometry1_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_2D_geometry2_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_2D_geometry3_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_2D_geometry4_t);
  
  return;
}



typedef struct{
  float *m, *h, *cellSize;
  float cst_x, cst_y, cst_z;
  int *periodic;
  int Nx, Ny, Nz, Nyz, Ntot;
} cpu_add_6NGBR_exchange_3D_geometry_arg;

void cpu_add_6NGBR_exchange_3D_geometry1_t(int id){

  cpu_add_6NGBR_exchange_3D_geometry_arg *arg = (cpu_add_6NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nyz);
  for (pH = arg->h + start, pM = arg->m + arg->Nyz + start; pH < arg->h + stop; pH++, pM++){    // add cst_x*M[x+1]
    *pH += arg->cst_x * (*pM);
  }
  if (arg->periodic[X]){
    init_start_stop (&start, &stop, id, arg->Nyz);
    for (pH = arg->h + arg->Ntot - arg->Nyz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)              // add cst_x*M[x=Nx] for periodic case
      *pH += arg->cst_x * (*pM);
  }
  return;
}
        
void cpu_add_6NGBR_exchange_3D_geometry2_t(int id){

  cpu_add_6NGBR_exchange_3D_geometry_arg *arg = (cpu_add_6NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nyz);
  for (pH = arg->h + arg->Nyz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                           // add cst_x*M[x-1] 
    *pH += arg->cst_x * (*pM);
  if (arg->periodic[X]){
    init_start_stop (&start, &stop, id, arg->Nyz);
    for (pH = arg->h + start, pM = arg->m + arg->Ntot - arg->Nyz + start; pH < arg->h + stop; pH++, pM++)             // add cst_x*M[x=-1] for periodic case
      *pH += arg->cst_x * (*pM);
  }

  return;
}

void cpu_add_6NGBR_exchange_3D_geometry3_t(int id){

  cpu_add_6NGBR_exchange_3D_geometry_arg *arg = (cpu_add_6NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Nx);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nyz, pM = arg->m + i*arg->Nyz + arg->Nz; pH<arg->h + (i+1)*arg->Nyz - arg->Nz; pH++, pM++)    // add cst_y*M[y+1]
      *pH += arg->cst_y * (*pM);
    if (arg->periodic[Y])                                                        
      for (pH = arg->h + (i+1)*arg->Nyz - arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)   // add cst_y*M[y=Ny] for periodic case
        *pH += arg->cst_y * (*pM);

    for (pH = arg->h + i*arg->Nyz + arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)         // add cst_y*M[y-1]
      *pH += arg->cst_y * (*pM);
    if (arg->periodic[Y])                                                        
      for (pH = arg->h + i*arg->Nyz, pM =  arg->m + (i+1)*arg->Nyz - arg->Nz; pH<arg->h + i*arg->Nyz + arg->Nz; pH++, pM++) // add cst_y*M[y=-1] for periodic case
        *pH += arg->cst_y * (*pM);
  }

  return;
}

void cpu_add_6NGBR_exchange_3D_geometry4_t(int id){

  cpu_add_6NGBR_exchange_3D_geometry_arg *arg = (cpu_add_6NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Nx*arg->Ny);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 1; pH<arg->h + (i+1)*arg->Nz - 1; pH++, pM++)    // add cst_z*M[z+1]
      *pH += arg->cst_z * (*pM);
    if (arg->periodic[Z])
      arg->h[(i+1)*arg->Nz-1] += arg->cst_z*arg->m[i*arg->Nz];                                          // add cst_z*M[z=Nz] for periodic case

    for (pH = arg->h + i*arg->Nz + 1, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)        // add cst_z*M[z+1]
      *pH += arg->cst_z * (*pM);
    if (arg->periodic[Z])
      arg->h[i*arg->Nz] += arg->cst_z*arg->m[(i+1)*arg->Nz - 1];                                        // add cst_z*M[z=-1] for periodic case
  }  

  return;
}

void cpu_add_6NGBR_exchange_3D_geometry5_t(int id){

  cpu_add_6NGBR_exchange_3D_geometry_arg *arg = (cpu_add_6NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot);
  for (pH = arg->h + start, pM = arg->m + start; pH < arg->h + stop; pH++, pM++)                              // add (cst_x + cst_y + cst_z) * M
    *pH -= 2.0f*(arg->cst_x + arg->cst_y + arg->cst_z) * (*pM);
  
  return;
}

void cpu_add_6NGBR_exchange_3D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize){

  cpu_add_6NGBR_exchange_3D_geometry_arg args;
  args.m = m;
  args.h = h;
  args.cellSize = cellSize;
  args.cst_x = 1.0f/cellSize[X]/cellSize[X];
  args.cst_y = 1.0f/cellSize[Y]/cellSize[Y];
  args.cst_z = 1.0f/cellSize[Z]/cellSize[Z];
  args.periodic = periodic;
  args.Nx = size[X];
  args.Ny = size[Y];
  args.Nz = size[Z];
  args.Nyz = size[Y]*size[Z];
  args.Ntot = size[X]*args.Nyz;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_6NGBR_exchange_3D_geometry1_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_3D_geometry2_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_3D_geometry3_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_3D_geometry4_t);
  thread_Wrapper(cpu_add_6NGBR_exchange_3D_geometry5_t);
  
  return;
}


typedef struct{
  float *m, *h, *cellSize;
  float cst0_y, cst1_y, cst2_y, cst0_z, cst1_z, cst2_z;
  int *periodic;
  int Ny, Nz, Ntot;
} cpu_add_12NGBR_exchange_2D_geometry_arg;

void cpu_add_12NGBR_exchange_2D_geometry1_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-2*arg->Nz);
  for (pH = arg->h + start, pM = arg->m + 2*arg->Nz + start; pH < arg->h + stop; pH++, pM++)             // add cst2_y*M[y+2]
    *pH += arg->cst2_y * (*pM);
  if (arg->periodic[Y]){
    init_start_stop (&start, &stop, id, 2*arg->Nz);
    for (pH = arg->h + arg->Ntot - 2*arg->Nz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)           // add cst2_y*M[y=Ny] and cst2_y*M[y=Ny+1] for periodic case
      *pH += arg->cst2_y * (*pM);
  }
  
  return;
}
  
void cpu_add_12NGBR_exchange_2D_geometry2_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nz);
  for (pH = arg->h + start, pM = arg->m + arg->Nz + start; pH < arg->h + stop; pH++, pM++)                 // add cst1_y*M[y+1]
    *pH += arg->cst1_y * (*pM);
  if (arg->periodic[Y]){
    init_start_stop (&start, &stop, id, arg->Nz);
    for (pH = arg->h + arg->Ntot - arg->Nz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)             // add cst1_y*M[y=Ny] for periodic case
      *pH += arg->cst1_y * (*pM);
  }
  
  return;
}
  
void cpu_add_12NGBR_exchange_2D_geometry3_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-2*arg->Nz);
  for (pH = arg->h + 2*arg->Nz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                    // add cst2_y*M[y-2] 
    *pH += arg->cst2_y * (*pM);
  if (arg->periodic[Y]){
    init_start_stop (&start, &stop, id, 2*arg->Nz);
    for (pH = arg->h + start, pM = arg->m + arg->Ntot - 2*arg->Nz + start; pH < arg->h + stop; pH++, pM++)           // add cst2_y*M[y=-2] and cst2_y*M[y=-1] for periodic case
      *pH += arg->cst2_y * (*pM);
  }
  
  return;
}

void cpu_add_12NGBR_exchange_2D_geometry4_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nz);
  for (pH = arg->h + arg->Nz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                      // add cst1_y*M[y-1] 
    *pH += arg->cst1_y * (*pM);
  if (arg->periodic[Y]){
    init_start_stop (&start, &stop, id, arg->Nz);
    for (pH = arg->h + start, pM = arg->m + arg->Ntot - arg->Nz + start; pH < arg->h + stop; pH++, pM++)               // add cst1_y*M[y=-1] for periodic case
      *pH += arg->cst1_y * (*pM);
  }
  
  return;
}
  
void cpu_add_12NGBR_exchange_2D_geometry5_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ny);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 2; pH<arg->h + (i+1)*arg->Nz-2; pH++, pM++)   // add cst2_z*M[z+2]
      *pH += arg->cst2_z * (*pM);
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 1; pH<arg->h + (i+1)*arg->Nz-1; pH++, pM++)   // add cst1_z*M[z+1]
      *pH += arg->cst1_z * (*pM);
    if (arg->periodic[Z]){                                                       
      arg->h[(i+1)*arg->Nz-2] += arg->cst2_z*arg->m[i*arg->Nz];                                      // add cst2_z*M[z=Nz] for periodic case
      arg->h[(i+1)*arg->Nz-1] += arg->cst2_z*arg->m[i*arg->Nz + 1];                                  // add cst2_z*M[z=Nz+1] for periodic case
      arg->h[(i+1)*arg->Nz-1] += arg->cst1_z*arg->m[i*arg->Nz];                                      // add cst1_z*M[z=Nz] for periodic case
    }
    
    for (pH = arg->h + i*arg->Nz + 2, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)     // add cst2_z*M[z-2]
      *pH += arg->cst2_z * (*pM);
    for (pH = arg->h + i*arg->Nz + 1, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)     // add cst1_z*M[z-1]
      *pH += arg->cst1_z * (*pM);
    if (arg->periodic[Z]){
      arg->h[i*arg->Nz] += arg->cst2_z*arg->m[(i+1)*arg->Nz-2];                                      // add cst2_z*M[z=-2] for periodic case
      arg->h[i*arg->Nz+1] += arg->cst2_z*arg->m[(i+1)*arg->Nz-1];                                    // add cst2_z*M[z=-1] for periodic case
      arg->h[i*arg->Nz] += arg->cst1_z*arg->m[(i+1)*arg->Nz-1];                                      // add cst1_z*M[z=-1] for periodic case
    }
  }
  
  return;
}

void cpu_add_12NGBR_exchange_2D_geometry6_t(int id){

  cpu_add_12NGBR_exchange_2D_geometry_arg *arg = (cpu_add_12NGBR_exchange_2D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot);
  for (pH = arg->h + start, pM = arg->m + start; pH < arg->h + stop; pH++, pM++)                           // add (cst0_y + cst0_z) * M
    *pH += (arg->cst0_y + arg->cst0_z) * (*pM);
  
  return;
}


void cpu_add_12NGBR_exchange_2D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize){

/*  float *pH, *pM;
  
  int Ny = size[Y];
  int Nz = size[Z];
  int Ntot = Ny*Nz;
  
  float cst0_y = -5.0f/2.0f/cellSize[Y]/cellSize[Y];
  float cst1_y = 4.0f/3.0f/cellSize[Y]/cellSize[Y];
  float cst2_y = -1.0f/12.0f/cellSize[Y]/cellSize[Y];
  float cst0_z = -5.0f/2.0f/cellSize[Z]/cellSize[Z];
  float cst1_z = 4.0f/3.0f/cellSize[Z]/cellSize[Z];
  float cst2_z = -1.0f/12.0f/cellSize[Z]/cellSize[Z];*/
  
  cpu_add_12NGBR_exchange_2D_geometry_arg args;
  args.m = m;
  args.h = h;
  args.cellSize = cellSize;
  args.cst0_y = -5.0f/2.0f/cellSize[Y]/cellSize[Y];
  args.cst1_y = 4.0f/3.0f/cellSize[Y]/cellSize[Y];
  args.cst2_y = -1.0f/12.0f/cellSize[Y]/cellSize[Y];
  args.cst0_z = -5.0f/2.0f/cellSize[Z]/cellSize[Z];
  args.cst1_z = 4.0f/3.0f/cellSize[Z]/cellSize[Z];
  args.cst2_z = -1.0f/12.0f/cellSize[Z]/cellSize[Z];
  args.periodic = periodic;
  args.Ny = size[Y];
  args.Nz = size[Z];
  args.Ntot = size[Y]*size[Z];

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry1_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry2_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry3_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry4_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry5_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_2D_geometry6_t);

//   for (pH = h, pM = m + 2*Nz; pH < h + Ntot - 2*Nz; pH++, pM++)             // add cst2_y*M[y+2]
//     *pH += cst2_y * (*pM);
//   for (pH = h, pM = m + Nz; pH < h + Ntot - Nz; pH++, pM++)                 // add cst1_y*M[y+1]
//     *pH += cst1_y * (*pM);
//   if (periodic[Y]){
//     for (pH = h + Ntot - 2*Nz, pM = m; pH < h + Ntot; pH++, pM++)           // add cst2_y*M[y=Ny] and cst2_y*M[y=Ny+1] for periodic case
//       *pH += cst2_y * (*pM);
//     for (pH = h + Ntot - Nz, pM = m; pH < h + Ntot; pH++, pM++)             // add cst1_y*M[y=Ny] for periodic case
//       *pH += cst1_y * (*pM);
//   }
//   
//   for (pH = h + 2*Nz, pM = m; pH < h + Ntot; pH++, pM++)                    // add cst2_y*M[y-2] 
//     *pH += cst2_y * (*pM);
//   for (pH = h + Nz, pM = m; pH < h + Ntot; pH++, pM++)                      // add cst1_y*M[y-1] 
//     *pH += cst1_y * (*pM);
//   if (periodic[Y]){
//     for (pH = h, pM = m + Ntot - 2*Nz; pH < h + 2*Nz; pH++, pM++)           // add cst2_y*M[y=-2] and cst2_y*M[y=-1] for periodic case
//       *pH += cst2_y * (*pM);
//     for (pH = h, pM = m + Ntot - Nz; pH < h + Nz; pH++, pM++)               // add cst1_y*M[y=-1] for periodic case
//       *pH += cst1_y * (*pM);
//   }
// 
//   
//   for (int i=0; i<Ny; i++){
//     for (pH = h + i*Nz, pM = m + i*Nz + 2; pH<h + (i+1)*Nz-2; pH++, pM++)   // add cst2_z*M[z+2]
//       *pH += cst2_z * (*pM);
//     for (pH = h + i*Nz, pM = m + i*Nz + 1; pH<h + (i+1)*Nz-1; pH++, pM++)   // add cst1_z*M[z+1]
//       *pH += cst1_z * (*pM);
//     if (periodic[Z]){                                                       
//       h[(i+1)*Nz-2] += cst2_z*m[i*Nz];                                      // add cst2_z*M[z=Nz] for periodic case
//       h[(i+1)*Nz-1] += cst2_z*m[i*Nz + 1];                                  // add cst2_z*M[z=Nz+1] for periodic case
//       h[(i+1)*Nz-1] += cst1_z*m[i*Nz];                                      // add cst1_z*M[z=Nz] for periodic case
//     }
//     
//     for (pH = h + i*Nz + 2, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)     // add cst2_z*M[z-2]
//       *pH += cst2_z * (*pM);
//     for (pH = h + i*Nz + 1, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)     // add cst1_z*M[z-1]
//       *pH += cst1_z * (*pM);
//     if (periodic[Z]){
//       h[i*Nz] += cst2_z*m[(i+1)*Nz-2];                                      // add cst2_z*M[z=-2] for periodic case
//       h[i*Nz+1] += cst2_z*m[(i+1)*Nz-1];                                    // add cst2_z*M[z=-1] for periodic case
//       h[i*Nz] += cst1_z*m[(i+1)*Nz-1];                                      // add cst1_z*M[z=-1] for periodic case
//     }
//   }
// 
//   for (pH = h, pM = m; pH < h + Ntot; pH++, pM++)                           // add (cst0_y + cst0_z) * M
//     *pH += (cst0_y + cst0_z) * (*pM);

  
  return;
}



typedef struct{
  float *m, *h, *cellSize;
  float cst0_x, cst1_x, cst2_x, cst0_y, cst1_y, cst2_y, cst0_z, cst1_z, cst2_z;
  int *periodic;
  int Nx, Ny, Nz, Nyz, Ntot;
} cpu_add_12NGBR_exchange_3D_geometry_arg;

void cpu_add_12NGBR_exchange_3D_geometry1_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-2*arg->Nyz);
  for (pH = arg->h + start, pM = arg->m + 2*arg->Nyz + start; pH < arg->h + stop; pH++, pM++)                      // add cst2_x*M[x+2]
    *pH += arg->cst2_x * (*pM);
  if (arg->periodic[X]){
    init_start_stop (&start, &stop, id, 2*arg->Nyz);
    for (pH = arg->h + arg->Ntot - 2*arg->Nyz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                      // add cst2_x*M[x=Nx] and cst2_x*M[x=Nx+1] for periodic case
      *pH += arg->cst2_x * (*pM);
  }
  
  return;
}

void cpu_add_12NGBR_exchange_3D_geometry2_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nyz);
  for (pH = arg->h + start, pM = arg->m + arg->Nyz + start; pH < arg->h + stop; pH++, pM++)                           // add cst1_x*M[x+1]
    *pH += arg->cst1_x * (*pM);
  if (arg->periodic[X])
    init_start_stop (&start, &stop, id, arg->Nyz);
    for (pH = arg->h + arg->Ntot - arg->Nyz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                        // add cst1_x*M[x=Nx] for periodic case
      *pH += arg->cst1_x * (*pM);
  
  return;
}
   
void cpu_add_12NGBR_exchange_3D_geometry3_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-2*arg->Nyz);
  for (pH = arg->h + 2*arg->Nyz + start, pM = arg->m + start; pH < arg->h + stop; pH++, pM++)                               // add cst2_x*M[x-2] 
    *pH += arg->cst2_x * (*pM);
  if (arg->periodic[X]){
    init_start_stop (&start, &stop, id, 2*arg->Nyz);
    for (pH = arg->h + start, pM = arg->m + arg->Ntot - 2*arg->Nyz + start; pH < arg->h + stop; pH++, pM++)                     // add cst2_x*M[x=-2] and cst2_x*M[x=-1] for periodic case
      *pH += arg->cst2_x * (*pM);
  }
  
  return;
}
  
void cpu_add_12NGBR_exchange_3D_geometry4_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot-arg->Nyz);
  for (pH = arg->h + arg->Nyz + start, pM = arg->m + start; pM < arg->m + stop; pH++, pM++)                                 // add cst1_x*M[x-1] 
    *pH += arg->cst1_x * (*pM);
  if (arg->periodic[X]){
    init_start_stop (&start, &stop, id, arg->Nyz);
    for (pH = arg->h + start, pM = arg->m + arg->Ntot - arg->Nyz + start; pH < arg->h + stop; pH++, pM++)                         // add cst1_x*M[x=-1] for periodic case
      *pH += arg->cst1_x * (*pM);
  }
   
  return;
}
 
void cpu_add_12NGBR_exchange_3D_geometry5_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Nx);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nyz, pM = arg->m + i*arg->Nyz + 2*arg->Nz; pH<arg->h + (i+1)*arg->Nyz - 2*arg->Nz; pH++, pM++)    // add cst2_y*M[y+2]
      *pH += arg->cst2_y * (*pM);
    for (pH = arg->h + i*arg->Nyz, pM = arg->m + i*arg->Nyz + arg->Nz; pH<arg->h + (i+1)*arg->Nyz - arg->Nz; pH++, pM++)        // add cst1_y*M[y+1]
      *pH += arg->cst1_y * (*pM);
    if (arg->periodic[Y]){                                                        
      for (pH = arg->h + (i+1)*arg->Nyz - 2*arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)     // add cst2_y*M[y=Ny] and cst2_y*M[y=Ny+1] for periodic case
        *pH += arg->cst2_y * (*pM);
      for (pH = arg->h + (i+1)*arg->Nyz - arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)       // add cst1_y*M[y=Ny] for periodic case
        *pH += arg->cst1_y * (*pM);
    }
    
    for (pH = arg->h + i*arg->Nyz + 2*arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)           // add cst2_y*M[y-2]
      *pH += arg->cst2_y * (*pM);
    for (pH = arg->h + i*arg->Nyz + arg->Nz, pM = arg->m + i*arg->Nyz; pH<arg->h + (i+1)*arg->Nyz; pH++, pM++)             // add cst1_y*M[y-1]
      *pH += arg->cst1_y * (*pM);
    if (arg->periodic[Y]){
      for (pH = arg->h + i*arg->Nyz, pM =  arg->m + (i+1)*arg->Nyz - 2*arg->Nz; pH<arg->h + i*arg->Nyz + 2*arg->Nz; pH++, pM++) // add cst2_y*M[y=-1] and cst2_y*M[y=-1] for periodic case
        *pH += arg->cst2_y * (*pM);
      for (pH = arg->h + i*arg->Nyz, pM =  arg->m + (i+1)*arg->Nyz - arg->Nz; pH<arg->h + i*arg->Nyz + arg->Nz; pH++, pM++)     // add cst1_y*M[y=-1] for periodic case
        *pH += arg->cst1_y * (*pM);
    }
  }
  
  return;
}

void cpu_add_12NGBR_exchange_3D_geometry6_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Nx*arg->Ny);
  for (int i=start; i<stop; i++){
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 2; pH<arg->h + (i+1)*arg->Nz - 2; pH++, pM++)             // add cst2_z*M[z+2]
      *pH += arg->cst2_z * (*pM);
    for (pH = arg->h + i*arg->Nz, pM = arg->m + i*arg->Nz + 1; pH<arg->h + (i+1)*arg->Nz - 1; pH++, pM++)             // add cst1_z*M[z+1]
      *pH += arg->cst1_z * (*pM);
    if (arg->periodic[Z]){
      arg->h[(i+1)*arg->Nz-2] += arg->cst2_z*arg->m[i*arg->Nz];                                                  // add cst2_z*M[z=Nz] for periodic case
      arg->h[(i+1)*arg->Nz-1] += arg->cst2_z*arg->m[i*arg->Nz+1];                                                // add cst2_z*M[z=Nz+1] for periodic case
      arg->h[(i+1)*arg->Nz-1] += arg->cst1_z*arg->m[i*arg->Nz];                                                  // add cst1_z*M[z=Nz] for periodic case
    }
    
    for (pH = arg->h + i*arg->Nz + 2, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)                 // add cst2_z*M[z+1]
      *pH += arg->cst2_z * (*pM);
    for (pH = arg->h + i*arg->Nz + 1, pM = arg->m + i*arg->Nz; pH<arg->h + (i+1)*arg->Nz; pH++, pM++)                 // add cst1_z*M[z+1]
      *pH += arg->cst1_z * (*pM);
    if (arg->periodic[Z]){
      arg->h[i*arg->Nz] += arg->cst2_z*arg->m[(i+1)*arg->Nz - 2];                                                // add cst2_z*M[z=-2] for periodic case
      arg->h[i*arg->Nz+1] += arg->cst2_z*arg->m[(i+1)*arg->Nz - 1];                                              // add cst2_z*M[z=-1] for periodic case
      arg->h[i*arg->Nz] += arg->cst1_z*arg->m[(i+1)*arg->Nz - 1];                                                // add cst1_z*M[z=-1] for periodic case
    }
  }  
  
  return;
}

void cpu_add_12NGBR_exchange_3D_geometry7_t(int id){

  cpu_add_12NGBR_exchange_3D_geometry_arg *arg = (cpu_add_12NGBR_exchange_3D_geometry_arg *) func_arg;
 
  float *pH, *pM;
  int start, stop;

  init_start_stop (&start, &stop, id, arg->Ntot);
  for (pH = arg->h + start, pM = arg->m + start; pH < arg->h + stop; pH++, pM++)                                       // add (cst0_x + cst0_y + cst0_z) * M
    *pH += (arg->cst0_x + arg->cst0_y + arg->cst0_z) * (*pM);
  
  return;
}
        
void cpu_add_12NGBR_exchange_3D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize){

  cpu_add_12NGBR_exchange_3D_geometry_arg args;
  args.m = m;
  args.h = h;
  args.cellSize = cellSize;
  args.cst0_x = -5.0f/2.0f/cellSize[X]/cellSize[X];
  args.cst1_x = 4.0f/3.0f/cellSize[X]/cellSize[X];
  args.cst2_x = -1.0f/12.0f/cellSize[X]/cellSize[X];
  args.cst0_y = -5.0f/2.0f/cellSize[Y]/cellSize[Y];
  args.cst1_y = 4.0f/3.0f/cellSize[Y]/cellSize[Y];
  args.cst2_y = -1.0f/12.0f/cellSize[Y]/cellSize[Y];
  args.cst0_z = -5.0f/2.0f/cellSize[Z]/cellSize[Z];
  args.cst1_z = 4.0f/3.0f/cellSize[Z]/cellSize[Z];
  args.cst2_z = -1.0f/12.0f/cellSize[Z]/cellSize[Z];
/*  args.cst0_x = 0.0f;
  args.cst1_x = 1.0f;
  args.cst2_x = 1.0f;
  args.cst0_y = 0.0f;
  args.cst1_y = 1.0f;
  args.cst2_y = 1.0f;
  args.cst0_z = 0.0f;
  args.cst1_z = 1.0f;
  args.cst2_z = 1.0f;*/
  args.periodic = periodic;
  args.Nx = size[X];
  args.Ny = size[Y];
  args.Nz = size[Z];
  args.Nyz = size[Y]*size[Z];
  args.Ntot = size[X]*args.Nyz;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry1_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry2_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry3_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry4_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry5_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry6_t);
  thread_Wrapper(cpu_add_12NGBR_exchange_3D_geometry7_t);

  return;
}




#ifdef __cplusplus
}
#endif