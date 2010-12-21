#include "cpu_init.h"
#include <sys/sysinfo.h>
#include "fftw3.h"
#include <omp.h>
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_init(){
  // set up multi-threaded fftw
    fftwf_init_threads();
    fftwf_plan_with_nthreads(get_nprocs_conf()); // automatically determines the available number of CPU's

    init_Threads(4);
      omp_set_num_threads(1);
}


#ifdef __cplusplus
}
#endif
