/**
 * @file
 *
 * This file implements the classical computation of the exchange field on gpu.
 * The procedure checks for every component of the field if the exchange contribution is allready added in the convolution.  
 * If not, the contribution is added here.  A distinction is made between 2D and 3D geometries.  The gpu computations 
 * require the use of shared memory of which the size is predefined.
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef GPU_EXCHANGE_H
#define GPU_EXCHANGE_H

#include "param.h"
#include "gputil.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Important: EXCH_BLOCK_X*EXCH_BLOCK_Y needs to be larger than (or equal to) 2*(EXCH_BLOCK_X + EXCH_BLOCK_Y + 2)!! 
 * Probably the optimal block sizes depend on the used hardware.
 */
#define EXCH_BLOCK_X 16
#define EXCH_BLOCK_Y 4
#define I_OFF (EXCH_BLOCK_X+2)*(EXCH_BLOCK_Y+2)
#define J_OFF (EXCH_BLOCK_X+2)


/**
 * Adds the exchange contribution in a classical way.
 */
void add_exchange (tensor *m,    ///> input magnetization tensor
                   tensor *h,    ///> output field tensor
                   param *p      ///> simulation parameters
                  );


void gpu_add_6NGBR_exchange_3D_geometry (float *m,     ///> input magnetization tensor
                                         float *h,     ///> output field tensor
                                         param *p      ///> simulation parameters
                                         );

__global__ void _gpu_add_6NGBR_exchange_3D_geometry(float *m,         ///> float array containing one component of the magnetization data
                                                    float *h,         ///> float array containing one component of the field data
                                                    int Nx,           ///> dimensions in x-direction
                                                    int Ny,           ///> dimensions in y-direction
                                                    int Nz,           ///> dimensions in z-direction
                                                    float cst_x,      ///> prefactor to acount for the contribution of neighboring cells in x-direction
                                                    float cst_y,      ///> prefactor to acount for the contribution of neighboring cells in y-direction
                                                    float cst_z,      ///> prefactor to acount for the contribution of neighboring cells in z-direction
                                                    float cst_xyz,    ///> prefactor to acount for the self distribution
                                                    int periodic_X,   ///> zero if not periodic in x-direction
                                                    int periodic_Y,   ///> zero if not periodic in y-direction 
                                                    int periodic_Z    ///> zero if not periodic in z-direction
                                                    );
                                  

void gpu_add_6NGBR_exchange_2D_geometry (float *m,     ///> float array containing one component of the magnetization data
                                         float *h,     ///> float array containing one component of the field data
                                         param *p      ///> simulation parameters
                                         );

__global__ void _gpu_add_6NGBR_exchange_2D_geometry(float *m, 
                                                    float *h,         ///> float array containing one component of the field data
                                                    int Ny,           ///> dimensions in y-direction
                                                    int Nz,           ///> dimensions in z-direction
                                                    float cst_y,      ///> prefactor to acount for the contribution of neighboring cells in y-direction
                                                    float cst_z,      ///> prefactor to acount for the contribution of neighboring cells in z-direction
                                                    float cst_yz,     ///> prefactor to acount for the self distribution
                                                    int periodic_Y,   ///> zero if not periodic in y-direction 
                                                    int periodic_Z    ///> zero if not periodic in z-direction
                                                    );
                                              
             
#ifdef __cplusplus
}
#endif
#endif