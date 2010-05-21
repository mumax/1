#include "stridedtensor.h"
#include "gputil.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void stride_test(){
  size_t width = 10;
  size_t height = 10;
  
  float* devPtr;
  size_t pitch;
  gpu_safe( cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height)) ;

  printf("pitch: %ud\n", pitch);
  
}


#ifdef __cplusplus
}
#endif


// cudaPitchedPtr Struct Reference
// Data Fields
// • size_t pitch
// Pitch of allocated memory in bytes.
// • void ∗ ptr
// Pointer to allocated memory.
// • size_t xsize
// Logical width of allocation in elements.
// • size_t ysize
// Logical height of allocation in elements.


// struct cudaPitchedPtr make_cudaPitchedPtr (void ∗ d, size_t p, size_t xsz, size_t ysz) [read]
// Returns a cudaPitchedPtr based on the speciﬁed input parameters d, p, xsz, and ysz.
// Parameters:
// d - Pointer to allocated memory
// p - Pitch of allocated memory in bytes
// xsz - Logical width of allocation in elements
// ysz - Logical height of allocation in elements
// Returns:
// cudaPitchedPtr speciﬁed by d, p, xsz, and ysz


//  cudaError_t cudaMallocPitch (void ∗∗ devPtr, size_t ∗ pitch, size_t width, size_t height)
// Allocates at least widthInBytes ∗ height bytes of linear memory on the device and returns in ∗devPtr a pointer
// to the allocated memory. The function may pad the allocation to ensure that corresponding pointers in any given row
// will continue to meet the alignment requirements for coalescing as the address is updated from row to row. The pitch
// returned in ∗pitch by cudaMallocPitch() is the width in bytes of the allocation. The intended usage of pitch is as
// a separate parameter of the allocation, used to compute addresses within the 2D array. Given the row and column of
// an array element of type T, the address is computed as:
// T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
// For allocations of 2D arrays, it is recommended that programmers consider performing pitch allocations using cud-
// aMallocPitch(). Due to pitch alignment restrictions in the hardware, this is especially true if the application will be
// performing 2D memory copies between different regions of device memory (whether linear memory or CUDA arrays).
// Parameters:
// devPtr - Pointer to allocated pitched device memory
// pitch - Pitch for allocation
// width - Requested pitched allocation width
// height - Requested pitched allocation height
// Returns:
// cudaSuccess, cudaErrorMemoryAllocation
// Note:
// Note that this function may also return error codes from previous, asynchronous launches.
