/**
 * @file
 * Initialization of the micromagnetic kernel for 3D simulations (possibly only 1 cell thickness).  
 * This kernel contains the demag and exchange (if wanted) contribution and is adapted to the 
 * FFT dimensions of gpufft.cu. The Kernel components are Fourier transformed and only the real 
 * parts are stored on the device. The correction factor 1/(dimensions FFTs) is included.
 * Parameter demagCoarse controls if the kernel is defined on a coarser discretization level.
 * Parameter exchInConv controls if exchange is included in the kernel
 * 
 * @author Ben Van de Wiele
 */
#ifndef GPU_MICROMAG3D_KERNEL_H
#define GPU_MICROMAG3D_KERNEL_H

#include "tensor.h"
#include "gputil.h"
#include "param.h"
//#include "gpufft2.h"
#include "gpu_fft.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

//______________________________________________________________________________ kernel initialization


/**
 * Returns the initialized micromagnetic 3D kernel.  The kernel is stored as a rank 2 tensor.
 * For a 3D simulation with thickness > 1 FD cell, the first rank contains the Kernel elements:
 * [xx, xy, xz, yy, yz, zz], for a 3D simulation with thickness = 1 FD cell (only possible in 
 * X-direction!) the first rank contains only the non-zero elements: [xx, yy, yz, zz].
 * The second rank contains the contiguous Fourier transformed data of each element.  For instants: 
 * the contiguous xy-data maps the x-component of the (Fourier transformed) magnetic field 
 * with the y-component of the (Fourier transformed) magnetization.
 * 
 * This function includes:
 * 		- computation of the elements of the Greens kernel components
 *		- Fourier transformation of the Greens kernel components
 *		- extraction of the real parts in the Fourier domain (complex parts are zero due to symmetry)
 * 
 * Demag, exchange (if wanted) as well as the correction factor 1/(dimensions FFTs) are included.
 */
tensor *gpu_micromag3d_kernel(param *p              ///< parameter list
                             );

/**
 * Initializes the Greens kernel elements, Fourier transforms the data and extracts 
 * the real parts from the data. (imaginary parts are zero in due to the symmetry)
 * The kernel is only stored at the device.
 */
void gpu_init_and_FFT_Greens_kernel_elements_micromag3d(tensor *dev_kernel,  			///< rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
                                                        int *kernelSize, 			    ///< Non-strided size of the kernel data
                                                        int *exchInConv,           ///< 3 ints, 1 means exchange is included in the kernel in the considered direction
                                                        float *FD_cell_size, 			///< 3 float, size of finite difference cell in X,Y,Z respectively
                                                        int *repetition, 					///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
                                                        float *dev_qd_P_10,  			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                        float *dev_qd_W_10,  			///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                        gpuFFT3dPlan* kernel_plan  /// FFT plan for the execution of the forward FFT of the kernel.
                                                        );

/**
 * Computes all elements of the Greens kernel component defined by 'co1, co2'.
 */
__global__ void _gpu_init_Greens_kernel_elements_micromag3d(float *dev_temp, 			///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                                            int Nkernel_X, 				///< Non-strided size of the kernel data (x-direction)
                                                            int Nkernel_Y, 				///< Non-strided size of the kernel data (y-direction) 
                                                            int Nkernel_Z,  				///< Non-strided size of the kernel data (z-direction) 
                                                            int Nkernel_storage_Z, ///< Strided size of the kernel data in z-direction
                                                            int exchInConv_X,      ///< 1 if exchange is to be included in the x-direction
                                                            int exchInConv_Y,      ///< 1 if exchange is to be included in the y-direction
                                                            int exchInConv_Z,      ///< 1 if exchange is to be included in the z-direction
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
__device__ float _gpu_get_Greens_element_micromag3d(int Nkernel_X, 					///< Non-strided size of the kernel data (x-direction)
                                                    int Nkernel_Y, 					///< Non-strided size of the kernel data (y-direction)
                                                    int Nkernel_Z, 					///< Non-strided size of the kernel data (z-direction)
                                                    int exchInConv_X,        ///< 1 if exchange is to be included in the x-direction
                                                    int exchInConv_Y,        ///< 1 if exchange is to be included in the y-direction
                                                    int exchInConv_Z,        ///< 1 if exchange is to be included in the z-direction
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
__global__ void _gpu_extract_real_parts_micromag3d(float *dev_kernel_array, 	///< pointer to the first kernel element of the considered tensor element
                                                   float *dev_temp  					///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                                   );
																				 
																				 

/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used for integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_on_gpu_micromag3d(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                   float *dev_qd_P_10,  ///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                   float *FD_cell_size  ///< 3 floats: the dimensions of the used FD cell, (X, Y, Z) respectively
                                                   );
																				 
/**
 *Get the quadrature points for integration between a and b
 */
void get_Quad_Points_micromag3d(float *gaussQP, 			///< float array containing the requested Gauss quadrature points
                                float *stdgaussQ,		  ///< standard Gauss quadrature points between -1 and +1
                                int qOrder, 					///< Gauss quadrature order
                                double a,  					  ///< integration lower bound
                                double b							///< integration upper bound
                                );

		    

#ifdef __cplusplus
}
#endif
#endif