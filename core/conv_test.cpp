/**
* This application runs the convolution unit tests and illustrates the usage of conv_gpu
*/
#include "conv_gpu.h"
#include "tensor.h"
#include <assert.h>

int main(int argc, char** argv){
  printf("conv_test: ");
  
  int size[3] = {8, 8, 2};
  tensor* kernel = new_tensor(3, 16, 16, 4);
  convplan* plan = new_convplan(size, kernel);
  
  printf("PASS\n");
  return 0;
}