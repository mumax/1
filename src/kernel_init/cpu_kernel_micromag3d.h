/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 * Initialization of the micromagnetic kernel elements for 3D simulations (possibly only 1 cell thickness).  
 * The small particle limit is incorporated i.e. averaging of the demag field of the considered FD cell.
 * 
 * @author Ben Van de Wiele
 */
#ifndef CPU_KERNEL_MICROMAG3D_H
#define CPU_KERNEL_MICROMAG3D_H

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Initializes the micromagnetic 3D kernel element defined by 'co1, co2'.
 * The kernel element is written to stdout in tensor format.
 */
void cpu_init_kernel_elements_micromag3d(int co1,               ///< defines the kernel element: e.g. Kxy has co1=0, co2=1
                                         int co2,               ///< defines the kernel element: e.g. Kxy has co1=0, co2=1
                                         int *kernelSize, 		  ///< Non-strided size of the kernel data
                                         float *cellSize,       ///< 3 float, size of finite difference cell in X,Y,Z respectively
                                         int *repetition  		  ///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)
                                         );

/**
 * Computes all elements of the micromagnetic 3D kernel component defined by 'co1, co2'.
 */
void _cpu_init_kernel_elements_micromag3d(float *data, 		      ///< pointer to the temporary memory space on the device to store all elements of a given Greens tensor component
                                          int *Nkernel,      	  ///< Non-strided size of the kernel data (x,y,z-direction) 
                                          int co1, 					 	  ///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                          int co2, 					    ///< co1 and co2 define the requested Greens tensor component: e.g. co1=0, co2=1 defines gxy
                                          float *cellSize,      ///< Size of the used FD cells
                                          int *repetition, 	 	  ///< 2*repetition[i]+1 is the number of periods considered the i-direction (repetition[i]=0 means no repetion in i direction)
                                          float *qd_P_10, 			///< float array (30 floats) containing the 10 Gauss quadrature points for X, Y and Z contiguously (on device)
                                          float *qd_W_10				///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                          );

/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used e.g. for integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_micromag3d(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
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