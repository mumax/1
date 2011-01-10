#include "gpu_local_contr2.h"
#include "gpu_mem.h"
#include "gpu_conf.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_add_local_fields_uniaxial(float* mx, float* my, float* mz,
                                              float* hx, float* hy, float* hz,
                                              float hext_x, float hext_y, float hext_z,
                                              float U0, float U1, float U2,
                                              int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i] += hext_x + mu * U0;
    hy[i] += hext_y + mu * U1;
    hz[i] += hext_z + mu * U2;
    
  }
}


__global__ void _gpu_add_external_field(float* hx, float* hy, float* hz,
                                        float hext_x, float hext_y, float hext_z,
                                        int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}


void gpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){


  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  /*
    Uniaxial anisotropy:
    H_anis = ( 2K_1 / (mu0 Ms) )  ( m . u ) u
    U := sqrt( 2K_1 / (mu0 Ms) )
    H_anis = (m . U) U
  */
  float U0, U1, U2;
  
  dim3 gridsize, blocksize;
  make1dconf(N, &gridsize, &blocksize);

  switch (anisType){
    default: abort();
    case ANIS_NONE:
       _gpu_add_external_field<<<gridsize, blocksize>>>(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
       break;
    case ANIS_UNIAXIAL:
      U0 = sqrt(2.0 * anisK[0]) * anisAxes[0];
      U1 = sqrt(2.0 * anisK[0]) * anisAxes[1];
      U2 = sqrt(2.0 * anisK[0]) * anisAxes[2];
	  //printf("anis: K, u, U: %f  %f,%f,%f %f,%f,%f \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2], U0, U1, U2);
      _gpu_add_local_fields_uniaxial<<<gridsize, blocksize>>>(mx, my, mz,
                                                             hx, hy, hz,
                                                             Hext[X], Hext[Y], Hext[Z],
                                                             U0, U1, U2, N);
      break;
  }
}

  /*                        
__global__ void _gpu_add_local_contr(float* mx, float* my, float* mz,
                                     float* hx, float* hy, float* hz,
                                     float Hax, float Hay, float Haz,
                                     int anisType, dev_par *p_dev, int N){
  
  int i = threadindex;

  if(i < N){
    
    if (anisType == NONE){
      hx[i] += Hax;
      hy[i] += Hay;
      hz[i] += Haz;
    }


    if (anisType == ANIS_UNIAXIAL){
      float projection = 2.0f*p_dev->anisK[0] * (mx[i]*p_dev->anisAxes[X] + my[i]*p_dev->anisAxes[Y] + mz[i]*p_dev->anisAxes[Z]);
      hx[i] += Hax + projection*p_dev->anisAxes[X];
      hy[i] += Hay + projection*p_dev->anisAxes[Y];
      hz[i] += Haz + projection*p_dev->anisAxes[Z];
    }
    

    if (anisType == ANIS_CUBIC){
        //projection of m on cubic anisotropy axes
      float a0 = mx[i]*p_dev->anisAxes[0] + my[i]*p_dev->anisAxes[1] + mz[i]*p_dev->anisAxes[2];
      float a1 = mx[i]*p_dev->anisAxes[3] + my[i]*p_dev->anisAxes[4] + mz[i]*p_dev->anisAxes[5];
      float a2 = mz[i]*p_dev->anisAxes[6] + my[i]*p_dev->anisAxes[7] + mz[i]*p_dev->anisAxes[8];
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = p_dev->anisK[0] * (a11+a22) * a0  +  p_dev->anisK[1] * a0  *a11 * a22;
      float dphi_1 = p_dev->anisK[0] * (a00+a22) * a1  +  p_dev->anisK[1] * a00 *a1  * a22;
      float dphi_2 = p_dev->anisK[0] * (a00+a11) * a2  +  p_dev->anisK[1] * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      hx[i] += Hax - dphi_0*p_dev->anisAxes[0] - dphi_1*p_dev->anisAxes[3] - dphi_2*p_dev->anisAxes[6];
      hy[i] += Hay - dphi_0*p_dev->anisAxes[1] - dphi_1*p_dev->anisAxes[4] - dphi_2*p_dev->anisAxes[7];
      hz[i] += Haz - dphi_0*p_dev->anisAxes[2] - dphi_1*p_dev->anisAxes[5] - dphi_2*p_dev->anisAxes[8];
    }

  }
  
  return;
}


void gpu_add_local_contr (float *m, float *h, int Ntot, float *Hext, int anisType, dev_par *p_dev){

  float *hx = h + X*Ntot;
  float *hy = h + Y*Ntot;
  float *hz = h + Z*Ntot;

  float *mx = m + X*Ntot;
  float *my = m + Y*Ntot;
  float *mz = m + Z*Ntot;

  dim3 gridsize, blocksize;
  make1dconf(Ntot, &gridsize, &blocksize);
  _gpu_add_local_contr<<<gridsize, blocksize>>>(mx, my, mz, hx, hy, hz, Hext[X], Hext[Y], Hext[Z], anisType,  p_dev, Ntot);

}
                            

dev_par* init_par_on_dev(int anisType, float *anisK, float *defAxes)  {
  
  dev_par *p_dev = (dev_par*) malloc(sizeof(dev_par));
  p_dev->anisK = NULL;
  p_dev->anisAxes = NULL;

    //for uniaxial anisotropy
  if (anisType == ANIS_UNIAXIAL){
    p_dev->anisK = new_gpu_array(1);
    printf("strength: %e\n", anisK[0]);
    p_dev->anisK[0] = anisK[0];
    printf("strength assigned\n");
    
    p_dev->anisAxes = new_gpu_array(3);  
    float length = sqrt(defAxes[X]*defAxes[X] + defAxes[Y]*defAxes[Y] + defAxes[Z]*defAxes[Z]);
    p_dev->anisAxes[X] = defAxes[X]/length;
    p_dev->anisAxes[Y] = defAxes[Y]/length;
    p_dev->anisAxes[Z] = defAxes[Z]/length;
  }

    //for cubic anisotropy
  if (anisType == ANIS_CUBIC){
    p_dev->anisK = new_gpu_array(2);
    p_dev->anisK[0] = anisK[0];
    p_dev->anisK[1] = anisK[1];
    
    p_dev->anisAxes = new_gpu_array(9);
    float phi   = defAxes[X];
    float theta = defAxes[Y];
    float psi   = defAxes[Z];
    p_dev->anisAxes[0] = cos(psi)*cos(phi)-cos(theta)*sin(phi)*sin(psi);
    p_dev->anisAxes[1] = cos(psi)*sin(phi)+cos(theta)*cos(phi)*sin(psi);
    p_dev->anisAxes[2] = sin(psi)*sin(theta);
    p_dev->anisAxes[3] = -sin(psi)*cos(phi)-cos(theta)*sin(phi)*cos(psi);
    p_dev->anisAxes[4] = -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
    p_dev->anisAxes[5] = cos(psi)*sin(theta);
    p_dev->anisAxes[6] = sin(theta)*sin(phi);
    p_dev->anisAxes[7] = -sin(theta)*cos(phi);
    p_dev->anisAxes[8] = cos(theta);
  }

  return(p_dev);
}               

void destroy_par_on_dev(dev_par *p_dev, int anisType){

  if (anisType != NONE){
    free_gpu_array(p_dev->anisK);
    free_gpu_array(p_dev->anisAxes);
  }
  
  free (p_dev);
  
  return;
}*/


#ifdef __cplusplus
}
#endif
