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
 * @author Arne Vansteenkiste
 */
#ifndef gpu_exch_h
#define gpu_exch_h

#ifdef __cplusplus
extern "C" {
#endif

/// Adds the exchange field to h
void gpu_add_exch(float* m,       ///< magnetization (all 3 components, contiguously)
                  float* h,       ///< effective field, hexch to be added.
                  int N0,         ///< length of each of the components of m, h (1/3 of m's total length)
                  int N1,
                  int N2,
                  int wrap0,         ///< != 0 means periodicity in that direction.
                  int wrap1,
                  int wrap2,
				  float cellsize0,
				  float cellsize1,
				  float cellsize2, 
				  int type);    ///< exchange type (number of neighbors): 4, 6, ...

#ifdef __cplusplus
}
#endif
#endif
