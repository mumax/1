#include "gpu_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaStream_t* gpu_streamBuffer;
int gpu_stream_uninitiated = 1;
int gpu_current_stream = -1;


cudaStream_t gpu_getstream(){
  if(gpu_stream_uninitiated){
    gpu_init_stream();
  }
  cudaStream_t stream = gpu_streamBuffer[gpu_current_stream];
  gpu_current_stream++;
  if(gpu_current_stream == MAXGPUSTREAMS){
    gpu_current_stream = 0;
  }
  return stream;
}


void gpu_init_stream(){
  gpu_streamBuffer = (cudaStream_t*)calloc(MAXGPUSTREAMS, sizeof(cudaStream_t));
  for(int i=0; i<MAXGPUSTREAMS; i++){
    cudaStreamCreate(&gpu_streamBuffer[i]);
  }
  gpu_stream_uninitiated = 0;
}

#ifdef __cplusplus
}
#endif
