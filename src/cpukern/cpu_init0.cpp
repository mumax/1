/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_init.h"
#include <sys/sysinfo.h>
#include "fftw3.h"
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

void cpu_init(){
  // set up multi-threaded fftw
    //fftwf_init_threads();
    //fftwf_plan_with_nthreads(get_nprocs_conf()); // automatically determines the available number of CPU's
 omp_set_num_threads(1);
}

#ifdef __cplusplus
}
#endif
