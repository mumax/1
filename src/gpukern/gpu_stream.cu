/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_stream.h"
#include <stdio.h>
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
  //printf("gpu_getstream():%ld\n", (long int)stream);
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
