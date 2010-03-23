/**
* This application runs the convolution unit tests and illustrates the usage of conv_gpu
*/
#include "conv_gpu.h"
#include "tensor.h"
#include <assert.h>

int main(int argc, char** argv){
  printf("conv_test: ");
  
  int N0 = 8, N1 = 8, N2 = 2;
  int* size = new int[3];
  size[0] = N0; size[1] = N1, size[2] = N2;
  
  tensor* m = new_tensor(4, 3, N0, N1, N2);
  tensor* h = new_tensor(4, 3, N0, N1, N2);
  
  for(int i=0; i<tensor_length(m); i++){
    m->list[i] = i/100.0;
  }
  
  tensor* kernel = new_tensor(5, 3, 3, 2*N0, 2*N1, 2*N2);
  *tensor_get(kernel, 5, 0, 0, 0, 0, 0) = 1.0;
  
  convplan* plan = new_convplan(size, kernel);

  //format_tensor(kernel, stdout);
  //format_tensor(m, stdout);
  
  conv_execute(plan, m, h);
  
  //format_tensor(h, stdout);
  
  delete_convplan(plan);

  printf("PASS\n");
  return 0;
}