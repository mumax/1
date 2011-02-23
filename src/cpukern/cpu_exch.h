/**
 * @author Ben Van de Wiele
 */
#ifndef cpu_exch_h
#define cpu_exch_h

#ifdef __cplusplus
extern "C" {
#endif

/// Adds the exchange field to h.  It is checked internally if exchange is already added for some or all components in the convolution.

void cpu_add_exch(float* m,           ///< magnetization (all 3 components, contiguously)
                  float* h,           ///< effective field, hexch to be added.
                  int *size,          ///< number of FD cells in each direction.
                  int *periodic,      ///< !=0 means periodicity in that direction.
                  int *exchInconv0,   ///< !=0 means exchange is computed in the convolution and no classical evaluation is required.
                  float *cellSize,    ///< cell size in the corresponding directions.
                  int type);          ///< exchange type: EXCH_6NGBR, EXCH_12NGBR.



/// Adds the 6 neighbor exchange contribution for a 3D geometry (size[X]>1).
void cpu_add_6NGBR_exchange_3D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );
                                         
                                         
/// Adds the 6 neighbor exchange contribution for a 2D geometry (size[X]==1).  
/// In this case, the 6 neighbors reduce to 4 neighbors.
void cpu_add_6NGBR_exchange_2D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );


/// Adds the 12 neighbor exchange contribution for a 3D geometry (size[X]>1).
void cpu_add_12NGBR_exchange_3D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );
                                         
                                         
/// Adds the 12 neighbor exchange contribution for a 2D geometry (size[X]==1).  
/// In this case, the 8 neighbors reduce to 4 neighbors.
void cpu_add_12NGBR_exchange_2D_geometry (float *m,          ///< magnetization (all 3 components, contiguously)
                                         float *h,          ///< effective field, hexch to be added.
                                         int *size,         ///< number of FD cells in each direction.
                                         int *periodic,     ///< !=0 means periodicity in that direction.
                                         float *cellSize    ///< cell size in the corresponding directions.
                                         );

#ifdef __cplusplus
}
#endif
#endif
