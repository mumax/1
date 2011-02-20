/**
 * @file
 * Initialization of the micromagnetic kernel for 3D simulations (possibly only 1 cell thickness).  
 * This kernel contains the demag and exchange (if wanted) contribution and is adapted to the 
 * FFT dimensions of gpufft.cu. The Kernel components are Fourier transformed and only the real 
 * parts are stored on the device. The correction factor 1/(dimensions FFTs) is included.
 * Parameter exchInConv controls if exchange is included in the kernel
 * 
 * @author Ben Van de Wiele
 */
#ifndef CPU_KERNEL_MICROMAG3D_H
#define CPU_KERNEL_MICROMAG3D_H

#ifdef __cplusplus
extern "C" {
#endif

//______________________________________________________________________________ kernel initialization


/**
 * Returns the initialized micromagnetic 3D kernel.  The kernel is stored as a rank 2 tensor.
 * For a 3D simulation with thickness > 1 FD cell, the first rank contains the Kernel elements:
 * [x, y, z, yy, yz, zz], for a 3D simulation with thickness = 1 FD cell (only possible in 
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
void cpu_kernel_micromag3d(int *kernelSize,     ///< Non-strided size of the kernel data
                           float *cellsize,     ///< 3 float, size of finite difference cell in X,Y,Z respectively
                           int exchType,        ///< int representing the used exchange type
                           int *exchInConv,     ///< 3 ints, 1 means exchange is included in the kernel in the considered direction
                           int *repetition      ///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
                           );

                             
/**
 * Initializes the Greens kernel elements, Fourier transforms the data and extracts 
 * the real parts from the data. (imaginary parts are zero due to the symmetry)
 * The kernel is only stored at the device.
 */
void cpu_init_and_FFT_Greens_kernel_elements_micromag3d(float *dev_kernel,  			///< float array: list of rank 2 tensor; rank 0: xx, xy, xz, yy, yz, zz parts of symmetrical Greens tensor, rank 1: all data of a Greens kernel component contiguously
                                                        int *kernelSize, 			    ///< Non-strided size of the kernel data
                                                        int exchType,              ///< int representing the used exchange type
                                                        int *exchInConv,           ///< 3 ints, 1 means exchange is included in the kernel in the considered direction (1st int ignored)
                                                        float *FD_cell_size,      ///< 3 float, size of finite difference cell in X,Y,Z respectively
                                                        int *repetition, 					///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
                                                        float *dev_qd_P_10,  			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                        float *dev_qd_W_10,  			///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                        cpuFFT3dPlan *kernel_plan  /// FFT plan for the execution of the forward FFT of the kernel.
                                                        );

/**
 * Computes all elements of the Greens kernel component defined by 'co1, co2'.
 */
void _cpu_init_Greens_kernel_elements_micromag3d(float *dev_temp, 		 ///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                                 int *Nkernel,      	 ///< Non-strided size of the kernel data (x,y,z-direction) 
                                                 int exchType,         ///< int representing the used exchange type
                                                 int *exchInConv,      ///< 1 if exchange is to be included in the x, y and z-direction
                                                 int co1, 						 ///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                 int co2, 						 ///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                 float *FD_cell_size,  ///< Size of the used FD cells
                                                 int *repetition, 	 	 ///< 2*repetition[i]+1 is the number of periods considered the i-direction (repetition[i]=0 means no repetion in i direction)
                                                 float *qd_P_10, 			 ///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                 float *qd_W_10				 ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                 );

/**
 * Extracts the 'size1' real numbers from the array 'dev_temp' and 
 * stores them in the 'dev_kernel_array' starting from 'dev_kernel_array[rank0*size1]'.
 */
void _cpu_extract_real_parts_micromag3d(float *dev_kernel_array, 	///< pointer to the first kernel element of the considered tensor element
                                        float *dev_temp,  				///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                        int N
                                        );


/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used for integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_micromag3d(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
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