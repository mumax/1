#include "cpu_init.h"
#include <sys/sysinfo.h>
#include "fftw3.h"
#include <omp.h>
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_init(int threads,      ///< number of threads to use, 0 means autodect the number of CPUs
              int options       ///< currently not used
              ){
  
    if (threads <= 0){ // automatically use maximum number of threads
      threads = get_nprocs_conf();
    }
  
    if( threads > 1){
      // set up multi-threaded fftw
      fftwf_init_threads();
      fftwf_plan_with_nthreads(threads);
    }

    // set up openMP
    omp_set_num_threads(threads);

    // set up Ben
    init_Threads(threads);
}


#ifdef __cplusplus
}
#endif
