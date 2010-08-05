#include "cpu_transpose.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

///@todo cleanup

///@internal kernel
// __global__ void _cpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
  //     // N0 <-> N2
  //     // i  <-> k
  //     int N3 = 2;
  //
  //     int i = blockIdx.x;
  //     int j = blockIdx.y;
  //     int k = threadIdx.x;
  //
  //     dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
  //     dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
  // }



void cpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){

  assert(source != dest);{ // must be out-of-place

  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;  ///@todo: should have new variable here!
  int N3 = 2;

  //   dim3 gridsize(N0, N1, 1); ///@todo generalize!
  //   dim3 blocksize(N2, 1, 1);
  //   cpu_checkconf(gridsize, blocksize);
  //   _cpu_transposeXZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  //   cudaThreadSynchronize();
  for(int i=0; i<N0; i++){
    for(int j=0; j<N1; j++){
      for(int k=0; k<N2; k++){
        dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
        dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
      }
    }
  }
  }
  /*  else{
  cpu_transposeXZ_complex_inplace(source, N0, N1, N2*2); ///@todo see above
}*/
}


// __global__ void _cpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
  //     // N1 <-> N2
  //     // j  <-> k
  //
  //     int N3 = 2;
  //
  //         int i = blockIdx.x;
  //     int j = blockIdx.y;
  //     int k = threadIdx.x;
  //
  // //      int index_dest = i*N2*N1*N3 + k*N1*N3 + j*N3;
  // //      int index_source = i*N1*N2*N3 + j*N2*N3 + k*N3;
  //
  //
  //     dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
  //     dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
  // /*    dest[index_dest + 0] = source[index_source + 0];
  //     dest[index_dest + 1] = source[index_source + 1];*/
  // }

  void cpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){

    assert(source != dest);{ // must be out-of-place

    // we treat the complex array as a N0 x N1 x N2 x 2 real array
    // after transposing it becomes N0 x N2 x N1 x 2
    N2 /= 2;
    int N3 = 2;

    //   dim3 gridsize(N0, N1, 1); ///@todo generalize!
    //   dim3 blocksize(N2, 1, 1);
    //   cpu_checkconf(gridsize, blocksize);
    //   _cpu_transposeYZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
    //   cudaThreadSynchronize();

    for(int i=0; i<N0; i++){
      for(int j=0; j<N1; j++){
        for(int k=0; k<N2; k++){
          dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
          dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
        }
      }
    }

    }
    /*  else{
    cpu_transposeYZ_complex_inplace(source, N0, N1, N2*2); ///@todo see above
  }*/
    //timer_stop("transposeYZ");
  }



  


#ifdef __cplusplus
}
#endif
