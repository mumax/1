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
 * @author Ben Van de Wiele
 */
#ifndef gpu_exch_h
#define gpu_exch_h

#ifdef __cplusplus
extern "C" {
#endif

/// Adds the exchange field to h.  It is checked internally if exchange is already added for some or all components in the convolution.

void gpu_add_exch(float* m,           ///< magnetization (all 3 components, contiguously)
                  float* h,           ///< effective field, hexch to be added.
                  int *size,          ///< number of FD cells in each direction.
                  int *periodic,      ///< !=0 means periodicity in that direction.
                  int *exchInconv0,   ///< !=0 means exchange is computed in the convolution and no classical evaluation is required.
                  float *cellSize,    ///< cell size in the corresponding directions.
                  int type);          ///< exchange type: EXCH_6NGBR, EXCH_12NGBR.



/// Adds the 6 neighbor exchange contribution for a 3D geometry (size[X]>1).
void gpu_add_6NGBR_exchange_3D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );
                                         
                                         
/// Adds the 6 neighbor exchange contribution for a 2D geometry (size[X]==1).  
/// In this case, the 6 neighbors reduce to 4 neighbors.
void gpu_add_6NGBR_exchange_2D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );

#ifdef __cplusplus
}
#endif
#endif
