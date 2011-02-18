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
				  int type);    ///< exchange type (number of neighbors): 4, 6, ...

#ifdef __cplusplus
}
#endif
#endif
