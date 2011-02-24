/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_init.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Not much setup is needed here, only the number of threads per block is optionally set
void gpu_init(int threads,  ///< number of threads per block, 0 means autoset
              int options   ///< currently not used
              ){
  gpu_setmaxthreads(threads);
}

void gpu_set_device(int devid){
  gpu_safe(cudaSetDevice(devid));
}

#ifdef __cplusplus
}
#endif
