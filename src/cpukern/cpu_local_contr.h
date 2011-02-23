/**
 * @file
 *
 * This file implements the addition of the local contributions to the effective field.  These are
 *    - Applied field
 *    - anisotropy field
 *
 *    @todo kubical anisotropy thermal field, magnetoelastic field
 *
 * @author Arne Vansteenkiste, Ben Van de Wiele
 *
 */
#ifndef cpu_local_contr_h
#define cpu_local_contr_h

#ifdef __cplusplus
extern "C" {
#endif


/// Adds the local contributions to the effective field.
void cpu_add_local_fields (float* m,              ///> magnetization data Mx, My, Mz contiguously
                           float* h,              ///> effective field data Hx, Hy, Hz contiguously
                           int N,                 ///> number of FD cells
                           float* Hext,           ///> 3 floats, externally applied field
                           int anisType,          ///> anisotropy type
                           float* anisK,          ///> anisotropy constants
                           float* anisAxes        ///> anisotropy axes
                           );

/// Adds the external field to the effective field
void cpu_add_external_field(float* hx,            ///> effective field data: X component
                            float* hy,            ///> effective field data: Y component    
                            float* hz,            ///> effective field data: Z component
                            float hext_x,         ///> externally applied field: X component
                            float hext_y,         ///> externally applied field: Y component 
                            float hext_z,         ///> externally applied field: Z component
                            int N                 ///> number of FD cells
                            );

/// Adds the external field and the uniaxial anisotropy field to the effective field
void cpu_add_local_fields_uniaxial(float *mx,     ///> magnetization data: X component
                                   float *my,     ///> magnetization data: Y component 
                                   float *mz,     ///> magnetization data: Z component
                                   float* hx,     ///> effective field data: X component
                                   float* hy,     ///> effective field data: Y component
                                   float* hz,     ///> effective field data: Z component
                                   float hext_x,  ///> externally applied field: X component
                                   float hext_y,  ///> externally applied field: Y component 
                                   float hext_z,  ///> externally applied field: Z component 
                                   float anisK,   ///> uniaxial anisotropy constant
                                   float U0,      ///> uniaxial anisotropy axis: UO component
                                   float U1,      ///> uniaxial anisotropy axis: U1 component 
                                   float U2,      ///> uniaxial anisotropy axis: U2 component
                                   int N
                                   );
#ifdef __cplusplus
}
#endif
#endif