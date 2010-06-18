/**
 * @file
 * Initialization of the micromagnetic kernel for 3D simulations (possibly only 1 cell thickness).  
 * This kernel contains the demag and exchange contribution and is adapted to the FFT dimensions 
 * of gpufft.cu. The Kernel components are Fourier transformed and only the real parts are stored 
 * on the device. The correction factor 1/(dimensions FFTs) is included.
 * 
 * @author Ben Van de Wiele
 */
#ifndef GPU_MICROMAG3D_KERNEL_H
#define GPU_MICROMAG3D_KERNEL_H

#include "tensor.h"
#include "gputil.h"
#include "param.h"
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
 * Returns the Initialized micromagnetic 3D kernel.  The kernel is stored as a rank 2 tensor:
 * The i+j element (j>=i) of kernel[i+j][k] contains the contigous Fourier transformed data 
 * connecting the (Fourier transformed) i-component of the magnetic field with the
 * (Fourier transformed) j-component of the magnetization.
 * 
 * This function includes:
 * 		- computation of the elements of the Greens kernel components
 *		- Fourier transformation of the Greens kernel components
 *		- in Fourier domain: extraction of the real parts (complex parts are zero due to symmetry)
 * 
 * Demag, exchange as well as the correction factor 1/(dimensions FFTs) are included.
 */
tensor *gpu_micromag3d_kernel(param *p              ///< parameter list
                             );

/**
 * Initializes the Greens kernel elements, Fourier transforms the data and extracts 
 * the real parts from the data. (imaginary parts are zero in due to the symmetry)
 * The kernel is only stored at the device.
 */
void gpu_init_and_FFT_Greens_kernel_elements(tensor *dev_kernel,  								///< rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
																						 int *demagKernelSize, 			  				///< Non-strided size of the kernel data
																						 float *FD_cell_size, 								///< 3 float, size of finite difference cell in X,Y,Z respectively
																						 int *repetition, 										///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
																						 float *dev_qd_P_10,  								///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																						 float *dev_qd_W_10,  								///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																						 gpu_plan3d_real_input* kernel_plan   /// FFT plan for the execution of the forward FFT of the kernel.
																						 );

/**
 * Computes all elements of the Greens kernel component defined by 'co1, co2'.
 */
__global__ void _gpu_init_Greens_kernel_elements(float *dev_temp, 			///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
																								 int Nkernel_X, 				///< Non-strided size of the kernel data (x-direction)
																								 int Nkernel_Y, 				///< Non-strided size of the kernel data (y-direction) 
																								 int Nkernel_Z,  				///< Non-strided size of the kernel data (z-direction) 
																								 int co1, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																								 int co2, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
																								 float FD_cell_size_X, 	///< Size of the used FD cells in the x direction
																				 				 float FD_cell_size_Y, 	///< Size of the used FD cells in the y direction 
																								 float FD_cell_size_Z,  ///< Size of the used FD cells in the z direction
																								 int repetition_X, 			///< 2*repetition_X+1 is the number of periods considered the x-direction (repetition_X=0 means no repetion in x direction)
																								 int repetition_Y, 			///< 2*repetition_Y+1 is the number of periods considered the y-direction (repetition_Y=0 means no repetion in y direction)
																								 int repetition_Z, 			///< 2*repetition_Z+1 is the number of periods considered the z-direction (repetition_Z=0 means no repetion in z direction)
																								 float *qd_P_10, 				///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																								 float *qd_W_10					///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																								 );

/**
 * Returns an element with coordinates [a,b,c] of the Greens kernel component defined by 'co1, co2'.
 */
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
																				 int repetition_X, 				///< 2*repetition_X+1 is the number of periods considered the x-direction (repetition_X=0 means no repetion in x direction)
																				 int repetition_Y, 				///< 2*repetition_Y+1 is the number of periods considered the y-direction (repetition_Y=0 means no repetion in y direction)
																				 int repetition_Z, 				///< 2*repetition_Z+1 is the number of periods considered the z-direction (repetition_Z=0 means no repetion in z direction)
																				 float *dev_qd_P_10,			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
																				 float *dev_qd_W_10				///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
																				 );


/**
 * Extracts the 'size1' real numbers from the array 'dev_temp' and 
 * stores them in the 'dev_kernel_array' starting from 'dev_kernel_array[rank0*size1]'.
 */
__global__ void _gpu_extract_real_parts(float *dev_kernel_array, 	///< rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
																				float *dev_temp, 					///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
																				int rank0,								///< int defining the rank 0 number: xx=0, xy=1, xz=2, etc.
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
void get_Quad_Points(float *gaussQP, 			///< float array containing the requested Gauss quadrature points
										 float *stdgaussQ,		///< standard Gauss quadrature points between -1 and +1
										 int qOrder, 					///< Gauss quadrature order
										 double a,  					///< integration lower bound
										 double b							///< integration upper bound
										 );

		    

#ifdef __cplusplus
}
#endif
#endif