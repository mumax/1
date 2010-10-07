#include "gpu_exchange.h"

#ifdef __cplusplus
extern "C" {
#endif



///> important: EXCH_BLOCK_X*EXCH_BLOCK_Y needs to be larger than (or equal to) 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2)!! 

void add_exchange (tensor *m, tensor *h, param *p){

  float *m_comp = NULL;
  float *h_comp = NULL;
  
  if(p->exchInConv[X]!=0 && p->exchInConv[Y]!=0 && p->exchInConv[Z]!=0)
    return;
  
  for (int i=0; i<3; i++)
    if (p->exchInConv[i]==0){
      m_comp = &m->list[i*m->len/3];
      h_comp = &h->list[i*m->len/3];
      
      switch (p->exchType){
        case EXCH_6NGBR:
          if (p->size[X] == 1)
            gpu_add_6NGBR_exchange_2D_geometry (m_comp, h_comp, p);
          else 
            gpu_add_6NGBR_exchange_3D_geometry (m_comp, h_comp, p);
          break;
        default:
          fprintf(stderr, "abort: no valid exchType %d\n", p->exchType);
          abort();
      }
    return;
    }

}




void gpu_add_6NGBR_exchange_3D_geometry (float *m, float *h, param *p){

  float cst_x = 1.0f/p->cellSize[X]/p->cellSize[X];
  float cst_y = 1.0f/p->cellSize[Y]/p->cellSize[Y];
  float cst_z = 1.0f/p->cellSize[Z]/p->cellSize[Z];
  float cst_xyz = -2.0f/p->cellSize[X]/p->cellSize[X] - 2.0f/p->cellSize[Y]/p->cellSize[Y] - 2.0f/p->cellSize[Z]/p->cellSize[Z];


  int bx = 1 + (p->size[Z]-1)/EXCH_BLOCK_X;    // a grid has blockdim.z=1, so we use the x-component
  int by = 1 + (p->size[Y]-1)/EXCH_BLOCK_Y;

  dim3 gridsize (bx, by);
  dim3 blocksize (EXCH_BLOCK_X, EXCH_BLOCK_Y);

<<<<<<< HEAD:src/dev_ben/gpu_exchange.cu
  _gpu_add_6NGBR_exchange_3D_geometry <<<gridsize, blocksize>>> (m, h, p->size[X], p->size[Y], p->size[Z], cst_x, cst_y, cst_z, cst_xyz, p->demagPeriodic[X], p->demagPeriodic[Y], p->demagPeriodic[Z]);
=======
  _gpu_add_exchange_3D_geometry <<<gridsize, blocksize>>> (m, h, p->size[X], p->size[Y], p->size[Z], cst_x, cst_y, cst_z, cst_xyz, p->demagPeriodic[X], p->demagPeriodic[Y], p->demagPeriodic[Z]);
>>>>>>> arne:src/core/gpu_exchange.cu
  gpu_sync();
  
  return;
}
  
__global__ void _gpu_add_6NGBR_exchange_3D_geometry(float *m, float *h, int Nx, int Ny, int Nz, float cst_x, float cst_y, float cst_z, float cst_xyz, int periodic_X, int periodic_Y, int periodic_Z){

  float *hptr, result;
  int i, j, k, ind, ind_h, indg, indg_h, active;

  int Nyz = Ny*Nz; int Nx_minus_1 = Nx-1; int Ny_minus_1 = Ny-1; int Nz_minus_1 = Nz-1;
  int cnt = threadIdx.y*EXCH_BLOCK_X + threadIdx.x;


// initialize indices for halo elements ------------------------------------------------
  int halo = cnt < 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2);
  if (halo) {
    if (threadIdx.y<2) {               // shared memory indices, y-halos (coalesced)
      j = threadIdx.y*(EXCH_BLOCK_Y+1) - 1;
      k = threadIdx.x;
    }
    else {                             // shared memory indices, z-halos (not coalesced)
      j =  cnt/2 - EXCH_BLOCK_X - 1;
      k = (cnt%2)*(EXCH_BLOCK_X+1) - 1;
    }

    ind_h  = I_OFF + (j+1)*J_OFF + k+1 ;

    j = blockIdx.y*EXCH_BLOCK_Y + j;
    k = blockIdx.x*EXCH_BLOCK_X + k;         //global indices
    if (periodic_Y && j==-1) j += Ny;
    if (periodic_Y && j==Ny) j -= Ny;
    if (periodic_Z && k==-1) k += Nz;
    if (periodic_Z && k==Nz) k -= Nz;
    indg_h = j*Nz + k;

    halo = (j>=0) && (j<Ny) && (k>=0) && (k<Nz);
  }
// -------------------------------------------------------------------------------------

// initialize indices for main block ---------------------------------------------------
  j    = threadIdx.y;                   // shared memory indices
  k    = threadIdx.x;
  ind  = I_OFF + (j+1)*J_OFF + k+1;

  j    = blockIdx.y*EXCH_BLOCK_Y + j;   // global indices
  k    = blockIdx.x*EXCH_BLOCK_X + k;
  indg = j*Nz + k;

  active = (j<Ny) && (k<Nz);
// -------------------------------------------------------------------------------------

// initialize shared memory ------------------------------------------------------------
  __shared__ float m_sh[3*I_OFF];

// if periodic_X: read last yz-plane of array ------------------------------------------
  if (periodic_X){
    if (active) 
      m_sh[ind  ] = m[indg + Nx_minus_1*Nyz];
    if (halo) 
      m_sh[ind_h] = m[indg_h + Nx_minus_1*Nyz];
  }
// -------------------------------------------------------------------------------------

// read first yz plane of array --------------------------------------------------------
  if (active) 
    m_sh[ind   + I_OFF] = m[indg];
  if (halo) 
    m_sh[ind_h + I_OFF] = m[indg_h];
// -------------------------------------------------------------------------------------


// perform the actual exchange computations --------------------------------------------
  for (i=0; i<Nx; i++) {

    // move two planes down and read in new plane i+1
     if (active) {
      hptr = h + indg;   // identical to hptr = &h[indg]
      indg += Nyz;
      m_sh[ind - I_OFF] = m_sh[ind];
      m_sh[ind]         = m_sh[ind + I_OFF];
      if (i<Nx_minus_1)
        m_sh[ind + I_OFF] = m[indg];
      else if (periodic_X!=0)
        m_sh[ind + I_OFF] = m[indg - Nx_minus_1*Nyz];
    }

    if (halo) {
      indg_h = indg_h + Nyz;
      m_sh[ind_h - I_OFF] = m_sh[ind_h];
      m_sh[ind_h]         = m_sh[ind_h + I_OFF];
      if (i<Nx_minus_1)
        m_sh[ind_h + I_OFF] = m[indg_h];
      else if (periodic_X!=0)
        m_sh[ind_h + I_OFF] = m[indg_h - Nx_minus_1*Nyz];
    }
    __syncthreads();

    if (active){
      result = cst_xyz * m_sh[ind];
     if (i>0          || periodic_X) result += cst_x*m_sh[ind - I_OFF];
     if (i<Nx_minus_1 || periodic_X) result += cst_x*m_sh[ind + I_OFF];
     if (j>0          || periodic_Y) result += cst_y*m_sh[ind - J_OFF];
     if (j<Ny_minus_1 || periodic_Y) result += cst_y*m_sh[ind + J_OFF];
     if (k>0          || periodic_Z) result += cst_z*m_sh[ind - 1];
     if (k<Nz_minus_1 || periodic_Z) result += cst_z*m_sh[ind + 1];
     *hptr += result;
    }
    __syncthreads();

  }

  return;
}
  



void gpu_add_6NGBR_exchange_2D_geometry (float *m, float *h, param *p){

  float cst_y = 1.0f/p->cellSize[Y]/p->cellSize[Y];
  float cst_z = 1.0f/p->cellSize[Z]/p->cellSize[Z];
  float cst_yz = -2.0f/p->cellSize[Y]/p->cellSize[Y] - 2.0f/p->cellSize[Z]/p->cellSize[Z];

  int bx = 1 + (p->size[Z]-1)/EXCH_BLOCK_X;    // a grid has blockdim.z=1, so we use the x-component
  int by = 1 + (p->size[Y]-1)/EXCH_BLOCK_Y;

  dim3 gridsize (bx, by);
  dim3 blocksize (EXCH_BLOCK_X, EXCH_BLOCK_Y);

<<<<<<< HEAD:src/dev_ben/gpu_exchange.cu
  _gpu_add_6NGBR_exchange_2D_geometry <<<gridsize, blocksize>>> (m, h, p->size[Y], p->size[Z], cst_y, cst_z, cst_yz, p->demagPeriodic[Y], p->demagPeriodic[Z]);
=======
  _gpu_add_exchange_2D_geometry <<<gridsize, blocksize>>> (m, h, p->size[Y], p->size[Z], cst_y, cst_z, cst_yz, p->demagPeriodic[Y], p->demagPeriodic[Z]);
>>>>>>> arne:src/core/gpu_exchange.cu
  gpu_sync();
  
  return;
}

__global__ void _gpu_add_6NGBR_exchange_2D_geometry(float *m, float *h, int Ny, int Nz, float cst_y, float cst_z, float cst_yz, int periodic_Y, int periodic_Z){

  float result;
  int j, k, ind, ind_h, indg, indg_h, active;

  int Ny_minus_1 = Ny-1; int Nz_minus_1 = Nz-1;
  int cnt = threadIdx.y*EXCH_BLOCK_X + threadIdx.x;


// initialize indices for halo elements ------------------------------------------------
  int halo = cnt < 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2);
  if (halo) {
    if (threadIdx.y<2) {               // shared memory indices, y-halos (coalesced)
      j = threadIdx.y*(EXCH_BLOCK_Y+1) - 1;
      k = threadIdx.x;
    }
    else {                             // shared memory indices, z-halos (not coalesced)
      j =  cnt/2 - EXCH_BLOCK_X - 1;
      k = (cnt%2)*(EXCH_BLOCK_X+1) - 1;
    }

    ind_h  = (j+1)*J_OFF + k+1;

    j = blockIdx.y*EXCH_BLOCK_Y + j;
    k = blockIdx.x*EXCH_BLOCK_X + k;         //global indices
    if (periodic_Y && j==-1) j += Ny;
    if (periodic_Y && j==Ny) j -= Ny;
    if (periodic_Z && k==-1) k += Nz;
    if (periodic_Z && k==Nz) k -= Nz;
    indg_h = j*Nz + k;

    halo = (j>=0) && (j<Ny) && (k>=0) && (k<Nz);
  }
// -------------------------------------------------------------------------------------

// initialize indices for main block ---------------------------------------------------
  j    = threadIdx.y;                   // shared memory indices
  k    = threadIdx.x;
  ind  = (j+1)*J_OFF + k+1;

  j    = blockIdx.y*EXCH_BLOCK_Y + j;   // global indices
  k    = blockIdx.x*EXCH_BLOCK_X + k;
  indg = j*Nz + k;

  active = (j<Ny) && (k<Nz);
// -------------------------------------------------------------------------------------

// initialize shared memory ------------------------------------------------------------
  __shared__ float m_sh[I_OFF];

// read the considered part of the plane of array --------------------------------------
  if (active) 
    m_sh[ind  ] = m[indg];
  if (halo) 
    m_sh[ind_h] = m[indg_h];
// -------------------------------------------------------------------------------------
  __syncthreads();
  
// perform the actual exchange computations --------------------------------------------
  if (active){
    result = cst_yz * m_sh[ind];
    if (j>0          || periodic_Y) result += cst_y*m_sh[ind - J_OFF];
    if (j<Ny_minus_1 || periodic_Y) result += cst_y*m_sh[ind + J_OFF];
    if (k>0          || periodic_Z) result += cst_z*m_sh[ind - 1];
    if (k<Nz_minus_1 || periodic_Z) result += cst_z*m_sh[ind + 1];
    h[indg] += result;
  }
  __syncthreads();

  return;
}
  
  
#ifdef __cplusplus
}
#endif