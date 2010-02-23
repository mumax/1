#include <stdio.h>

int main(){

  int i;

  float* ah;
  float* bh;
  float* ad;
  float* bd;

  int n = 10;
  int nbytes = n*sizeof(float);

  ah = (float*)malloc(nbytes);
  bh = (float*)malloc(nbytes);

  for(i=0; i<n; i++){
    ah[i] = float(i);
  }

  for(i=0; i<n; i++){
    printf("%f\n", ah[i]);
  }

  cudaMalloc((void**)&ad, nbytes);
  cudaMalloc((void**)&bd, nbytes);

  // ! first dest, than source !
  cudaMemcpy(ad, ah, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, ad, nbytes, cudaMemcpyDeviceToDevice);
  cudaMemcpy(bh, bd, nbytes, cudaMemcpyDeviceToHost);

  for(i=0; i<n; i++){
    printf("%f\n", bh[i]);
  }

  free(ah);
  free(bh);
  cudaFree(ad);
  cudaFree(bd);

  return 0;
}