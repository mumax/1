#include "cpu_exch.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_addExch (tensor *m, tensor *h, param *p){

 p->exchInConv[X] = 0;   ///> for code checking!  should be deleted
 p->exchInConv[Y] = 0;   ///> for code checking!  should be deleted
 p->exchInConv[Z] = 0;   ///> for code checking!  should be deleted

  if(p->exchInConv[X]!=0 && p->exchInConv[Y]!=0 && p->exchInConv[Z]!=0)
    return;

  else{
      // allocate arrays on cpu to store 1 component
    float *m_comp = (float *) calloc(m->len/3, sizeof(float));
    float *h_comp = (float *) calloc(m->len/3, sizeof(float));

    for (int i=0; i<3; i++)
      if (p->exchInConv[i]==0){
           //copy the considered component to cpu
        memcpy_from_gpu(&m->list[i*m->len/3], m_comp, m->len/3);
        memcpy_from_gpu(&h->list[i*m->len/3], h_comp, m->len/3);

          // perform the actual exchange computations for the considered component
        if (p->size[X] == 1)
          cpu_addExch_2D_geometry (m_comp, h_comp, p);    /// @todo make a gpu_addExch_2D_geometry
        else 
          cpu_addExch_3D_geometry (m_comp, h_comp, p);    /// @todo make a gpu_addExch_3D_geometry    

          // copy the result back to the device
        memcpy_to_gpu(h_comp, &h->list[i*m->len/3], h->len/3);

      }

    free(m_comp);
    free(h_comp);

    return;
  }
}


void cpu_addExch_2D_geometry (float *m, float *h, param *p){

  float *pH, *pM;
  
  int Ny = p->size[Y];
  int Nz = p->size[Z];
  int Ntot = Ny*Nz;
  
  float cst_y = 1.0f/p->cellSize[Y]/p->cellSize[Y];
  
  for (pH = h, pM = m + Nz; pH < h + Ntot - Nz; pH++, pM++)     // add cst_y*M[y+1]
    *pH += cst_y * (*pM);
  if (p->demagPeriodic[Y])
    for (pH = h + Ntot - Nz, pM = m; pH < h + Ntot; pH++, pM++) // add cst_y*M[y=Ny] for periodic case
      *pH += cst_y * (*pM);

  for (pH = h + Nz, pM = m; pH < h + Ntot; pH++, pM++)          // add cst_y*M[y-1] 
    *pH += cst_y * (*pM);
  if (p->demagPeriodic[Y])
    for (pH = h, pM = m + Ntot - Nz; pH < h + Nz; pH++, pM++)  // add cst_y*M[y=-1] for periodic case
      *pH += cst_y * (*pM);


  float cst_z = 1.0f/p->cellSize[Z]/p->cellSize[Z];
  
  for (int i=0; i<Ny; i++){
    for (pH = h + i*Nz, pM = m + i*Nz + 1; pH<h + (i+1)*Nz-1; pH++, pM++)   // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (p->demagPeriodic[Z])                                                // add cst_z*M[z=Nz] for periodic case
      h[(i+1)*Nz-1] += cst_z*m[i*Nz];
    for (pH = h + i*Nz + 1, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)     // add cst_z*M[z-1]
      *pH += cst_z * (*pM);
    if (p->demagPeriodic[Z])                                                // add cst_z*M[z=-1] for periodic case
      h[i*Nz] += cst_z*m[(i+1)*Nz-1];
  }

  for (pH = h, pM = m; pH < h + Ntot; pH++, pM++)             // substract (cst_y + cst_z) * M
    *pH -= 2.0f*(cst_y + cst_z) * (*pM);

  
  return;
}

void cpu_addExch_3D_geometry (float *m, float *h, param *p){

  float *pH, *pM;
  
  int Nx = p->size[X];
  int Ny = p->size[Y];
  int Nz = p->size[Z];
  int Nyz = Ny*Nz;
  int Ntot = Nyz*Nx;
  int cnt=0;

  float cst_x = 1.0f/p->cellSize[X]/p->cellSize[X];

  for (pH = h, pM = m + Nyz; pH < h + Ntot - Nyz; pH++, pM++){    // add cst_x*M[x+1]
    *pH += cst_x * (*pM);
  }
  if (p->demagPeriodic[X])
    for (pH = h + Ntot - Nyz, pM = m; pH < h + Ntot; pH++, pM++) // add cst_x*M[x=Nx] for periodic case
      *pH += cst_x * (*pM);

  for (pH = h + Nyz, pM = m; pH < h + Ntot; pH++, pM++)           // add cst_x*M[x-1] 
    *pH += cst_x * (*pM);
  if (p->demagPeriodic[X])
    for (pH = h, pM = m + Ntot - Nyz; pH < h + Nyz; pH++, pM++)   // add cst_x*M[x=-1] for periodic case
      *pH += cst_x * (*pM);


  float cst_y = 1.0f/p->cellSize[Y]/p->cellSize[Y];

  for (int i=0; i<Nx; i++){
    for (pH = h + i*Nyz, pM = m + i*Nyz + Nz; pH<h + (i+1)*Nyz - Nz; pH++, pM++)    // add cst_y*M[y+1]
      *pH += cst_y * (*pM);
    if (p->demagPeriodic[Y])                                                        
      for (pH = h + (i+1)*Nyz - Nz, pM = m + i*Nyz; pH<h + (i+1)*Nyz; pH++, pM++)   // add cst_y*M[y=Ny] for periodic case
        *pH += cst_y * (*pM);

    for (pH = h + i*Nyz + Nz, pM = m + i*Nyz; pH<h + (i+1)*Nyz; pH++, pM++)         // add cst_y*M[y-1]
      *pH += cst_y * (*pM);
    if (p->demagPeriodic[Y])                                                        
      for (pH = h + i*Nyz, pM =  m + (i+1)*Nyz - Nz; pH<h + i*Nyz + Nz; pH++, pM++) // add cst_y*M[y=-1] for periodic case
        *pH += cst_y * (*pM);
  }

  float cst_z = 1.0f/p->cellSize[Z]/p->cellSize[Z];

  for (int i=0; i<Nx*Ny; i++){
    for (pH = h + i*Nz, pM = m + i*Nz + 1; pH<h + (i+1)*Nz - 1; pH++, pM++)    // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (p->demagPeriodic[Z])
      h[(i+1)*Nz-1] += cst_z*m[i*Nz];                                          // add cst_z*M[z=Nz] for periodic case

    for (pH = h + i*Nz + 1, pM = m + i*Nz; pH<h + (i+1)*Nz; pH++, pM++)        // add cst_z*M[z+1]
      *pH += cst_z * (*pM);
    if (p->demagPeriodic[Z])
      h[i*Nz] += cst_z*m[(i+1)*Nz - 1];                                        // add cst_z*M[z=-1] for periodic case
  }  

  for (pH = h, pM = m; pH < h + Ntot; pH++, pM++)             // substract (cst_x + cst_y + cst_z) * M
    *pH -= 2.0f*(cst_x + cst_y + cst_z) * (*pM);
  

  return;
}


#ifdef __cplusplus
}
#endif