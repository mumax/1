/**
 * @file
 * Initialization of the Greens function kernel adapted to the FFT dimensions of gpufft.cu.
 * 
 * @todo introduce a symmetrical tensor
 *
 * @author Ben Van de Wiele
 */
#ifndef GPUKERNEL1_H
#define GPUKERNEL1_H

#include "tensor.h"
#include "gputil.h"
#include <cufft.h>
#include "gpufft.h"

#ifdef __cplusplus
extern "C" {
#endif

//______________________________________________________________________________ kernel initialization


/**
 * Initializes the Greens function kernel.  The kernel is stored as a rank 3 tensor:
 * The i,j element of kernel[i][j][k] contains the contigous Fourier transformed data 
 * connecting the (Fourier transformed) i-component of the magnetic field with the
 * (Fourier transformed) j-component of the magnetization.
 * The size is addapted to the gpu-memory.
 */

void init_Greens_kernel1(tensor* kernel,  		///< rank 3 tensor containing all contigous Greens function data
											 	 int N0, 							///< size of simulation in x-direction
												 int N1, 							///< size of simulation in y-direction
												 int N2, 							///< size of simulation in z-direction
												 int *zero_pad, 			///< 3 ints, should be 1 or 0, meaning zero-padding or no zero-padding in X,Y,Z respectively
												 float *FD_cell_size	///< 3 float, size of finite difference cell in X,Y,Z respectively
												);

												
/**
 *get the quadrature points for integration between a and b
 */
void get_Quad_Points(float *gaussQP, 					///< float array containing the requested Gauss quadrature points
										 float *stdgaussQ,				///< standard Gauss quadrature points between -1 and +1
										 int qOrder, 							///< Gauss quadrature order
										 float a,  								///< integration lower bound
										 float b									///< integration upper bound
										 );

		    

#ifdef __cplusplus
}
#endif
#endif