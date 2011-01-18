/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_init_h
#define cpu_init_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Does the necessary initialization before the CPU backend can be used
 */
void cpu_init(int threads,      ///< number of threads to use, 0 means autodect the number of CPUs
              int options       ///< currently not used
              );

int cpu_maxthreads();

#ifdef __cplusplus
}
#endif
#endif
