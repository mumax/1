#include "libfft.h"
#include "../libtensor/libtensor.h"

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <assert.h>

using namespace std;

double sqr(double r){
  return r*r;
}

int main(int argc, char** argv){

    fft_init();
    int N0 = 2, N1 = 2, N2 = 4;
    
    tensor* orig = new_tensor(3, N0, N1, N2); // a 3-dimensional block of size 16x16x4
    tensor* transf = new_tensor(3, N0, N1, N2+2); // transformed data must be 1 complex number (two floats) larger in its last dimension.
    tensor* back = new_tensor(3, N0, N1, N2); // forward+backward transformed data;
    
    void* fw_plan = fft_init_forward(orig->size[0], orig->size[1], orig->size[2], orig->list, transf->list);
    void* bw_plan = fft_init_backward(orig->size[0], orig->size[1], orig->size[2], transf->list, back->list);
    
    for(int i=0; i<tensor_length(orig); i++){
      orig->list[i] = i;//(rand() % 10000) / 10000.0 + 0.01;
    }
    //orig->list[0] = 1.0;
    
    //print_tensor(orig);
    
    fft_execute(fw_plan);
    fft_execute(bw_plan);
    
    //print_tensor(transf);
    
    //print_tensor(back);
    
    double rms_error = 0.0;
    int N = tensor_length(orig);
    for(int i=0; i<tensor_length(orig); i++){
      rms_error += sqr(orig->list[i] - back->list[i] / N);
    }
    rms_error = sqrt(rms_error);
    
    cout << "FFT error: " << rms_error << endl;
    
    //assert(rms_error < 1E-5);
    
    fft_finalize();
}