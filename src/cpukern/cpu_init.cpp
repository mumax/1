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
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

int fftw_strategy = FFTW_MEASURE;
#define PATIENT_FLAG 1


int _cpu_maxthreads = 0;

void cpu_setmaxthreads(int max){
  if(max <= 0){
    _cpu_maxthreads = get_nprocs_conf();
  } else {
    _cpu_maxthreads = max;
  }
}

int cpu_maxthreads(){
  if (_cpu_maxthreads == 0){
    _cpu_maxthreads = get_nprocs_conf();
  }
  return _cpu_maxthreads;
}


void cpu_init(int threads,      ///< number of threads to use, 0 means autodect the number of CPUs
              int options       ///< bitwise OR of flags: PATIENT, ...
              ){
  
    cpu_setmaxthreads(threads);
    threads = cpu_maxthreads();
    
    if( threads > 1){
      // set up multi-threaded fftw
      fftwf_init_threads();
//       fftwf_plan_with_nthreads(threads);
    }

//     // set up openMP
//     omp_set_num_threads(threads);

    // set up Ben
    init_Threads(threads);

/*    // set up options
    if (options & PATIENT_FLAG){
      fftw_strategy = FFTW_PATIENT;
    }*/
}


#ifdef __cplusplus
}
#endif
