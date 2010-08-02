/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_normalize_h
#define cpu_normalize_h

#ifdef __cplusplus
extern "C" {
#endif


/// Normalizes the magnetization (or any other vector field) to norm one
void cpu_normalize_uniform(float* m,    ///< magnetization to normalize (contiguous: x, y, z components)
                           int N        ///< @warning lenght of ONE component, the total length of m is thus 3*N
                           );

/// Normalizes the magnetization (or any other vector field) to a space-dependent value
void cpu_normalize_map(float* m,        ///< magnetization to normalize (contiguous: x, y, z components)
                       float* map,      ///< space-dependent norm, length N
                       int N            ///< @warning lenght of ONE component, the total length of m is thus 3*N
                       );

#ifdef __cplusplus
}
#endif
#endif
