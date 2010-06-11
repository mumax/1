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
#include "assert.h"
#include "timer.h"
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

//______________________________________________________________________________ kernel initialization


/**
 * Initializes the Greens function kernel.  The kernel is stored as a rank 2 tensor:
 * The i+j element (j>=i) of kernel[i+j][k] contains the contigous Fourier transformed data 
 * connecting the (Fourier transformed) i-component of the magnetic field with the
 * (Fourier transformed) j-component of the magnetization.
 * The size is addapted to the gpu-memory.
 * 
 * This function includes:
 * 		- computation of the elements of the Greens kernel components
 *		- Fourier transformation of the Greens kernel components
 *		- in Fourier domain: extraction of the real parts (complex parts are zero due to symmetry)
 *
 * note that the correction factor 1/(dimensions FFTs) is already included.
 */

void gpu_init_Greens_kernel1(tensor* dev_kernel,  ///< rank 2 tensor (rank 1: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor) containing all contiguous Greens function data
														 int N0, 							///< size of simulation in x-direction
														 int N1, 							///< size of simulation in y-direction
														 int N2, 							///< size of simulation in z-direction
														 int *zero_pad, 			///< 3 ints, should be 1 or 0, meaning zero-padding or no zero-padding in X,Y,Z respectively
														 int *repetition,			///< 3 ints, for periodicity: e.g. 2*repetition[0]-1 is the number of periods considered the x-direction
														 float *FD_cell_size	///< 3 float, size of finite difference cell in X,Y,Z respectively
														 );

/**
 * Initializes the Greens kernel elements, Fourier transforms the data and extracts 
 * the real parts from the data. (imaginary parts are zero in due to the symmetry)
 * The kernel is only stored at the device.
 */

void gpu_init_and_FFT_Greens_kernel_elements(tensor *dev_kernel,  ///< rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
																						 int *Nkernel, 			  ///< Non-strided size of the kernel data
																						 float *FD_cell_size, ///< 3 float, size of finite difference cell in X,Y,Z respectively
																						 float cst, 					///< constant factor preceeding the expression of the magnetostatic field 
																						 int *repetition, 		///< 3 ints, for periodicity: e.g. 2*repetition[0]-1 is the number of periods considered the x-direction ([1,1,1] means no periodic repetition)
																						 float *dev_qd_P_10,  ///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																						 float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																						 gpu_plan3d_real_input* kernel_plan   /// FFT plan for the execution of the forward FFT of the kernel.
																						 );

	 
__global__ void _gpu_init_Greens_kernel_elements(float *dev_temp, 			///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
																								 int Nkernel_X, 				///< Non-strided size of the kernel data (x-direction)
																								 int Nkernel_Y, 				///< Non-strided size of the kernel data (y-direction) 
																								 int Nkernel_Z,  				///< Non-strided size of the kernel data (z-direction) 
																								 int co1, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																								 int co2, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																								 float FD_cell_size_X, 	///< Size of the used FD cells in the x direction
																				 				 float FD_cell_size_Y, 	///< Size of the used FD cells in the y direction 
																								 float FD_cell_size_Z,  ///< Size of the used FD cells in the z direction
																								 float cst, 						///< constant preceeding all Greens tensor elements (includes the correction for the dimensions of the FFTs in the convolution!)
																								 int repetition_X, 			///< 2*repetition_X-1 is the number of periods considered the x-direction (repetition_X=1 means no repetion in x direction)
																								 int repetition_Y, 			///< 2*repetition_Y-1 is the number of periods considered the y-direction (repetition_Y=1 means no repetion in y direction)
																								 int repetition_Z, 			///< 2*repetition_Z-1 is the number of periods considered the z-direction (repetition_Z=1 means no repetion in z direction)
																								 float *qd_P_10, 				///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																								 float *qd_W_10					///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																								 );


__device__ float _gpu_get_Greens_element(int Nkernel_X, 					///< Non-strided size of the kernel data (x-direction)
																				 int Nkernel_Y, 					///< Non-strided size of the kernel data (y-direction)
																				 int Nkernel_Z, 					///< Non-strided size of the kernel data (z-direction)
																				 int co1, 								///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																				 int co2, 								///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																				 int a, 									///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
																				 int b,  									///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
																				 int c,  									///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
																				 float FD_cell_size_X,  	///< Size of the used FD cells in the x direction
																				 float FD_cell_size_Y,  	///< Size of the used FD cells in the y direction
																				 float FD_cell_size_Z,  	///< Size of the used FD cells in the z direction
																				 float cst, 							///< constant preceeding all Greens tensor elements (includes the correction for the dimensions of the FFTs in the convolution!)
																				 int repetition_X, 				///< 2*repetition_X-1 is the number of periods considered the x-direction (repetition_X=1 means no repetion in x direction)
																				 int repetition_Y, 				///< 2*repetition_Y-1 is the number of periods considered the y-direction (repetition_Y=1 means no repetion in y direction)
																				 int repetition_Z, 				///< 2*repetition_Z-1 is the number of periods considered the z-direction (repetition_Z=1 means no repetion in z direction)
																				 float *dev_qd_P_10,			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																				 float *dev_qd_W_10				///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																				 );


/**
 * Extracts the 'size1' real numbers from the array 'dev_temp' and 
 * stores them in the 'dev_kernel_array' starting from 'dev_kernel_array[rang0*size1]'.
 */
__global__ void _gpu_extract_real_parts(float *dev_kernel_array, 	///< rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
																				float *dev_temp, 					///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
																				int rang0,								///< int defining the rank 0 number: xx=0, xy=1, xz=2, etc.
																				int size1									///< length of a Fourier transformed Greens kernel component (only real parts).
																				);
																				 
																				 

/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used for integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_on_gpu(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																				float *dev_qd_P_10,  ///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																				float *FD_cell_size  ///< 3 floats: the dimensions of the used FD cell, (X, Y, Z) respectively
																				);
																				 
/**
 *Get the quadrature points for integration between a and b
 */
void get_Quad_Points(float *gaussQP, 					///< float array containing the requested Gauss quadrature points
										 float *stdgaussQ,				///< standard Gauss quadrature points between -1 and +1
										 int qOrder, 							///< Gauss quadrature order
										 double a,  							///< integration lower bound
										 double b									///< integration upper bound
										 );

		    

#ifdef __cplusplus
}
#endif
#endif