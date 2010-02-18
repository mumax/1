/**
 * This program runs the Unittests for libtensor and illustrates its functions.
 */

#include "libtensor.h"
#include <assert.h>

int main(int argc, char** argv){
    int* buffer = (int*) calloc(10, sizeof(int));
    tensor* a, *b, *c, *d;
    int i,j;
    
    // make a 3x4 matrix
    buffer[0] = 3;
    buffer[1] = 4;
    a = new_tensor(2, buffer);
    
    // this is how you get the rank:
    assert(a->rank == 2);
    
    // the size gets copied, so can safely overwrite buffer without affecting the tensor:
    buffer[0] = 0;
    buffer[1] = 0;
    assert(a->size[0] == 3);
    assert(a->size[1] == 4);
    
    // the data is initialized to zero:
    for(i=0; i < a->size[0]; i++){
      for(j=0; j < a->size[1]; j++){
	buffer[0] = i;
	buffer[1] = j;
	assert(*tensor_elem(a, buffer) == 0.);
      }
    }
    delete_tensor(a);
    
    
    
    // there are easy-to-use constructors for tensors of low rank
    
    // the rank 0 tensor is a bit trivial, but included for completeness
    b = new_tensor0();		// this is shorthand for new_tensor(0, NULL);
    assert(b->rank == 0);
    assert(b->size == NULL);
    
    // this is just a list of length 100
    //c = new_tensor1(100);
    
    return 0;
}