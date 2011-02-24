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
 * Initialization of the micromagnetic kernel for 2D simulations with invariance considered in 
 * the x-direction of all microstructural properties and all fields.  
 * The small particle limit is incorporated i.e. averaging of the demag field over the considered FD cell.
 * 
 * @author Ben Van de Wiele
 */
#ifndef GPU_KERNEL_MICROMAG2D_H
#define GPU_KERNEL_MICROMAG2D_H

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Initializes the micromagnetic 2D kernel element defined by 'co1, co2'.
 * The kernel element is written to stdout in tensor format.
 */
void gpu_init_kernel_elements_micromag2d(int co1,                   ///< defines the kernel element: e.g. Kyz has co1=1, co2=2
                                         int co2,                   ///< defines the kernel element: e.g. Kyz has co1=1, co2=2
                                         int *kernelSize,           ///< Non-strided size of the kernel data
                                         float *cellSize,           ///< 3 float, size of finite difference cell in X,Y,Z respectively (1st float ignored)
                                         int *repetition            ///< 3 ints, for periodicity: e.g. 2*repetition[0]+1 is the number of periods considered the x-direction ([0,0,0] means no periodic repetition)  (1st int ignored)
                                         );


/**
 * Initialization of the Gauss quadrature points and quadrature weights to 
 * be used for e.g. integration over the FD cell faces.  
 * A ten points Gauss quadrature formula is used. 
 * The obtained quadrature weights and points are copied to the device.
 */
void initialize_Gauss_quadrature_on_gpu_micromag2d(float *dev_qd_W_10,  ///< float array (10 floats) containing the 10 Gauss quadrature weights (on device)
                                                   float *dev_qd_P_10,  ///< float array (20 floats) containing the 10 Gauss quadrature points for Y and Z contiguously (on device)
                                                   float *cellSize      ///< 3 floats: the dimensions of the used FD cell, (X, Y, Z) respectively
                                                   );
                                         
/**
 *Get the quadrature points for integration between a and b
 */
void get_Quad_Points_micromag2d(float *gaussQP,      ///< float array containing the requested Gauss quadrature points
                                float *stdgaussQ,    ///< standard Gauss quadrature points between -1 and +1
                                int qOrder,          ///< Gauss quadrature order
                                double a,            ///< integration lower bound
                                double b             ///< integration upper bound
                                );

        

#ifdef __cplusplus
}
#endif
#endif