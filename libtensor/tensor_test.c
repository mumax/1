/**
 * This program runs the Unittests for libtensor and illustrates its functions.
 */

#include "libtensor.h"

int main(int argc, char** argv){
    int* size_buffer = (int*) calloc(10, sizeof(int));
    tensor* a;
    
    size_buffer[0] = 3;
    size_buffer[1] = 4;
    a = new_tensor(2, size_buffer);
    
    return 0;
}