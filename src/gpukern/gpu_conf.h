/**
 * @file
 * Create and check CUDA thread launch configurations
 *
 * @todo Needs to be completely re-worked, with macros to define the indices etc.
 *
 * @author Arne Vansteenkiste
 */
#ifndef gpu_conf_h
#define gpu_conf_h

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Macro for 1D index "i" in a CUDA kernel.
 @code
  i = threadindex;
 @endcode
 */
#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/**
 * Macro for integer division, but rounded UP
 */
#define divUp(x, y) ( (((x)-1)/(y)) +1 )
// It's almost like LISP ;-)


/*
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @deprecated use check3dconf(), which uses the actual device properties
 */
// void gpu_checkconf(dim3 gridsize, ///< 3D size of the thread grid
//            dim3 blocksize ///< 3D size of the trhead blocks on the grid
//            );

/*
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @deprecated use check1dconf(), which uses the actual device properties
 */    
// void gpu_checkconf_int(int gridsize, ///< 1D size of the thread grid
//                int blocksize ///< 1D size of the trhead blocks on the grid
//                );
               
/**
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */
void check3dconf(dim3 gridsize, ///< 3D size of the thread grid
           dim3 blocksize ///< 3D size of the trhead blocks on the grid
           );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */    
void check1dconf(int gridsize, ///< 1D size of the thread grid
               int blocksize ///< 1D size of the trhead blocks on the grid
               );
               

//______________________________________________________________________________________ make conf

/*
 * Makes a 3D thread configuration suited for a float array of size N0 x N1 x N2.
 * The returned configuration will:
 *  - span the entire N0 x N1 x N2 array
 *  - have the largest valid block size that fits in the N0 x N1 x N2 array
 *  - be valid
 *
 * @todo works only up to N2 = 512 
 * @see make1dconf()
 *
 * Example:
 * @code
  dim3 gridSize, blockSize;
  make3dconf(N0, N1, N2, &gridSize, &blockSize);
  mykernel<<<gridSize, blockSize>>>(arrrrgh);
  
  __global__ void mykernel(aaarghs){
    
    int i = ((blockIdx.x * blockDim.x) + threadIdx.x)
    int j = ((blockIdx.y * blockDim.y) + threadIdx.y)
    int k = ((blockIdx.z * blockDim.z) + threadIdx.z)
    
    ...
  }
 * @endcode
 */
// void make3dconf(int N0,     ///< size of 3D array to span
//         int N1,     ///< size of 3D array to span
//         int N2,     ///< size of 3D array to span
//         dim3* gridSize, ///< grid size is returned here
//         dim3* blockSize ///< block size is returned here
//         );
        
/**
 * Makes a 1D thread configuration suited for a float array of size N
 * The returned configuration will span at least the entire array but
 * can be larger. Your kernel should use the threadindex macro to
 * get the index "i", and check if it is smaller than the size "N" of
 * the array it is meant to iterate over.
 * @see make3dconf() threadindex
 *
 * Example:
 * @code
 * dim3 gridSize, blockSize;
 * make1dconf(arraySize, &gridSize, &blockSize);
 * mykernel<<<gridSize, blockSize>>>(arrrrghs, arraySize);

   __global__ void mykernel(int aargs, int N){
    int i = threadindex; //built-in macro
    if(i < N){  // check if i is in range!
      do_work();
    }
   }
 * @endcode
 */
void make1dconf(int N,          ///< size of array to span (number of floats)
                dim3* gridSize,  ///< grid size is returned here
                dim3* blockSize  ///< block size is returned here
                );



#ifdef __cplusplus
}
#endif
#endif
