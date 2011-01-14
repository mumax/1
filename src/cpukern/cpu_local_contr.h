/**
 * @file
 *
 * This file implements the addition of the local contributions to the effective field.  These are
 *    - Applied field
 *    - anisotropy field
 *
 *    @todo thermal field, magnetoelastic field
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef cpu_local_contr_h
#define cpu_local_contr_h

#ifdef __cplusplus
extern "C" {
#endif


/// Adds the local contributions to the effective field.
void cpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes);



#ifdef __cplusplus
}
#endif
#endif