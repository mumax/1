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
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_init_h
#define gpu_init_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Does the necessary initialization before the GPU backend can be used
 */
void gpu_init(int threads,  ///< number of threads per block, 0 means autoset
              int options   ///< currently not used
              );

/**
 * Selects a GPU when more than one is present
 */
void gpu_set_device(int devid);

#ifdef __cplusplus
}
#endif
#endif
