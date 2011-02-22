#include "cpu_exch.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_add_exch (float *m, float *h, int *size, int *periodic, int *exchInconv0, float *cellSize, int type){

  if(exchInConv[X]!=0 && exchInConv[Y]!=0 && exchInConv[Z]!=0)
    return;

  int N = size[X]*size[Y]*size[Z];
  for (int i=0; i<3; i++)
    if (exchInConv[i]==0){
      m_comp = &m[i*N/3];
      h_comp = &h[i*N/3];
      
      switch (type){
        case EXCH_6NGBR:
          if size[X] == 1)
            cpu_add_6NGBR_exchange_2D_geometry (m_comp, h_comp, int *size, int *periodic, float *cellSize);
          else 
            cpu_add_6NGBR_exchange_3D_geometry (m_comp, h_comp, int *size, int *periodic, float *cellSize);
          break;
/*        case EXCH_12NGBR:
          if size[X] == 1)
            gpu_add_12NGBR_exchange_2D_geometry (m_comp, h_comp, p);
          else 
            gpu_add_12NGBR_exchange_3D_geometry (m_comp, h_comp, p);
          break;*/
        default:
          fprintf(stderr, "abort: no valid exchType %d\n", type);
          abort();
      }
    }
    
  return;
}


void cpu_add_6NGBR_exchange_2D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize)){

  float *pH, *pM;
  
  int Ny = size[Y];
  int Nz = size[Z];
  int Ntot = Ny*Nz;
  
  float cst_y = 1.0f/cellSize[Y]/cellSize[Y];
  
  for (pH = h, pM = m + Nz; pH < h + Ntot - Nz; pH++, pM++)     // add cst_y*M[y+1]
    *pH += cst_y * (*pM);
  if (periodic[Y])
    for (pH = h + Ntot - Nz, pM = m; pH < h + Ntot; pH++, pM++) // add cst_y*M[y=Ny] for periodic case
      *pH += cst_y * (*pM);

  for (pH = h + Nz, pM = m; pH < h + Ntot; pH++, pM++)          // add cst_y*M[y-1] 
    *pH += cst_y * (*pM);
  if (periodic[Y])
    for (pH = h, pM = m + Ntot - Nz; pH < h + Nz; pH++, pM++)   // add cst_y*M[y=-1] for periodic case
      *pH += cst_y * (*pM);


  float cst_z = 1.0f/cellSize[Z]/cellSize[Z];
  
  for (int i=0; i<Ny; i++){
    for (pH = h + i*Nz, pM = m + i*Nz + 1; pH<h + (i+1)*Nz-1; pH++, pM++)   // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (periodic[Z])                                                        // add cst_z*M[z=Nz] for periodic case
      h[(i+1)*Nz-1] += cst_z*m[i*Nz];
    for (pH = h + i*Nz + 1, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)     // add cst_z*M[z-1]
      *pH += cst_z * (*pM);
    if (periodic[Z])                                                        // add cst_z*M[z=-1] for periodic case
      h[i*Nz] += cst_z*m[(i+1)*Nz-1];
  }

  for (pH = h, pM = m; pH < h + Ntot; pH++, pM++)             // substract (cst_y + cst_z) * M
    *pH -= 2.0f*(cst_y + cst_z) * (*pM);

  
  return;
}

void cpu_add_6NGBR_exchange_3D_geometry (float *m, float *h, int *size, int *periodic, float *cellSize){

  float *pH, *pM;
  
  int Nx = size[X];
  int Ny = size[Y];
  int Nz = size[Z];
  int Nyz = Ny*Nz;
  int Ntot = Nyz*Nx;
  int cnt=0;

  float cst_x = 1.0f/cellSize[X]/cellSize[X];

  for (pH = h, pM = m + Nyz; pH < h + Ntot - Nyz; pH++, pM++){    // add cst_x*M[x+1]
    *pH += cst_x * (*pM);
  }
  if (periodic[X])
    for (pH = h + Ntot - Nyz, pM = m; pH < h + Ntot; pH++, pM++)  // add cst_x*M[x=Nx] for periodic case
      *pH += cst_x * (*pM);

  for (pH = h + Nyz, pM = m; pH < h + Ntot; pH++, pM++)           // add cst_x*M[x-1] 
    *pH += cst_x * (*pM);
  if (periodic[X])
    for (pH = h, pM = m + Ntot - Nyz; pH < h + Nyz; pH++, pM++)   // add cst_x*M[x=-1] for periodic case
      *pH += cst_x * (*pM);


  float cst_y = 1.0f/cellSize[Y]/cellSize[Y];

  for (int i=0; i<Nx; i++){
    for (pH = h + i*Nyz, pM = m + i*Nyz + Nz; pH<h + (i+1)*Nyz - Nz; pH++, pM++)    // add cst_y*M[y+1]
      *pH += cst_y * (*pM);
    if (periodic[Y])                                                        
      for (pH = h + (i+1)*Nyz - Nz, pM = m + i*Nyz; pH<h + (i+1)*Nyz; pH++, pM++)   // add cst_y*M[y=Ny] for periodic case
        *pH += cst_y * (*pM);

    for (pH = h + i*Nyz + Nz, pM = m + i*Nyz; pH<h + (i+1)*Nyz; pH++, pM++)         // add cst_y*M[y-1]
      *pH += cst_y * (*pM);
    if (periodic[Y])                                                        
      for (pH = h + i*Nyz, pM =  m + (i+1)*Nyz - Nz; pH<h + i*Nyz + Nz; pH++, pM++) // add cst_y*M[y=-1] for periodic case
        *pH += cst_y * (*pM);
  }

  float cst_z = 1.0f/cellSize[Z]/cellSize[Z];

  for (int i=0; i<Nx*Ny; i++){
    for (pH = h + i*Nz, pM = m + i*Nz + 1; pH<h + (i+1)*Nz - 1; pH++, pM++)    // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (periodic[Z])
      h[(i+1)*Nz-1] += cst_z*m[i*Nz];                                          // add cst_z*M[z=Nz] for periodic case

    for (pH = h + i*Nz + 1, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)        // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (periodic[Z])
      h[i*Nz] += cst_z*m[(i+1)*Nz - 1];                                        // add cst_z*M[z=-1] for periodic case
  }  

  for (pH = h, pM = m; pH < h + Ntot; pH++, pM++)             // substract (cst_x + cst_y + cst_z) * M
    *pH -= 2.0f*(cst_x + cst_y + cst_z) * (*pM);
  

  return;
}


#ifdef __cplusplus
}
#endif