/**
 * @file
 * Initialization of the micromagnetic kernel elements for 3D simulations (possibly only 1 cell thickness).  
 * The small particle limit is incorporated i.e. everaging of the demag field over the considered FD cell.
 * 
 * @author Ben Van de Wiele
 */
#ifndef GPU_KERNEL_MICROMAG3D_H
#define GPU_KERNEL_MICROMAG3D_H

#ifdef __cplusplus
extern "C" {
#endif

//______________________________________________________________________________ kernel initialization


                            
/**
 * Initializes the micromagnetic 3D kernel elements defined by 'co1, co2'
 * The kernel is written to stdout in tensor format.
 */
void gpu_init_kernel_elements_micromag3d(int co1,                   ///< defines the kernel element: e.g. Kxy has co1=0, co2=1
                                         int co2,                   ///< defines the kernel element: e.g. Kxy has co1=0, co2=1
                                         int *kernelSize, 			    ///< Non-strided size of the kernel data
                                         float *cellSize, 			    ///< 3 floats, size of finite difference cell in X,Y,Z respectively
                                         int *repetition  					///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
                                         );

/**
 * Computes all elements of the micromagnetic 3D kernel component defined by 'co1, co2'.
 */
__global__ void _gpu_init_kernel_elements_micromag3d(float *dev_temp, 			///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                                     int Nkernel_X, 				///< Non-strided size of the kernel data (x-direction)
                                                     int Nkernel_Y, 				///< Non-strided size of the kernel data (y-direction) 
                                                     int Nkernel_Z,  			///< Non-strided size of the kernel data (z-direction) 
                                                     int co1, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                     int co2, 							///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                     float cellSize_X,     ///< Size of the used FD cells in the x direction
                                                     float cellSize_Y,     ///< Size of the used FD cells in the y direction 
                                                     float cellSize_Z,     ///< Size of the used FD cells in the z direction
                                                     int repetition_X, 		///< 2*repetition_X+1 is the number of periods considered the x-direction (repetition_X=0 means no repetion in x direction)
                                                     int repetition_Y, 		///< 2*repetition_Y+1 is the number of periods considered the y-direction (repetition_Y=0 means no repetion in y direction)
                                                     int repetition_Z, 		///< 2*repetition_Z+1 is the number of periods considered the z-direction (repetition_Z=0 means no repetion in z direction)
                                                     float *qd_P_10, 			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                     float *qd_W_10				///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                     );

/**
 * Returns an element with coordinates [a,b,c] of the Greens kernel component defined by 'co1, co2'.
 */
__device__ float _gpu_get_kernel_element_micromag3d(int Nkernel_X, 					///< Non-strided size of the kernel data (x-direction)
                                                    int Nkernel_Y, 					///< Non-strided size of the kernel data (y-direction)
                                                    int Nkernel_Z, 					///< Non-strided size of the kernel data (z-direction)
                                                    int co1, 								///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                    int co2, 								///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                                    int a, 									///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
                                                    int b,  								///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
                                                    int c,  								///< [a,b,c] defines the cartesian vector pointing to the source FD cell to the receiver FD cell (in units FD cell size) 
                                                    float cellSize_X,  	    ///< Size of the used FD cells in the x direction
                                                    float cellSize_Y,  	    ///< Size of the used FD cells in the y direction
                                                    float cellSize_Z,  	    ///< Size of the used FD cells in the z direction
                                                    int repetition_X, 			///< 2*repetition_X+1 is the number of periods considered the x-direction (repetition_X=0 means no repetion in x direction)
                                                    int repetition_Y, 			///< 2*repetition_Y+1 is the number of periods considered the y-direction (repetition_Y=0 means no repetion in y direction)
                                                    int repetition_Z, 			///< 2*repetition_Z+1 is the number of periods considered the z-direction (repetition_Z=0 means no repetion in z direction)
                                                    float *dev_qd_P_10,			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                    float *dev_qd_W_10			///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                    );


/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used for integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_on_gpu_micromag3d(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                   float *dev_qd_P_10,  ///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                                   float *cellSize      ///< 3 floats: the dimensions of the used FD cell, (X, Y, Z) respectively
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