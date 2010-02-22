#include "libfft.h"
#include "../libtensor/libtensor.h"

int main(int argc, char** argv){

    tensor* orig = new_tensor(3, 16, 16, 4); // a 3-dimensional block of size 16x16x4
    tensor* transf = new_tensor(3, 16, 16, 4+2); // transformed data must be 1 complex number (two floats) larger in its last dimension.
    tensor* back = new_tensor(3, 16, 16, 4); // forward+backward transformed data;
    
    void* fw_plan = fft_init_forward(orig->size[0], orig->size[1], orig->size[2], orig->list, transf->list);
    void* bw_plan = fft_init_backward(orig->size[0], orig->size[1], orig->size[2], transf->list, back->list);
}