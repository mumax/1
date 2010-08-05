/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_zeropad_h
#define cpu_zeropad_h

#ifdef __cplusplus
extern "C" {
#endif


void cpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2);       ///< dest size


void cpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2);



#ifdef __cplusplus
}
#endif
#endif
