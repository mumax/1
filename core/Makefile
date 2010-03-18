include ../common.mk

CPPFILES=\
	tensor.cpp\


CUDA_LIBS=-L /usr/local/cuda/lib/libcudart.so -L /usr/local/cuda/lib/libcufft.so
NVCCFLAGS =--compiler-options -fPIC --compiler-options -g

all:	*.h $(CPPFILES) *.cu
	$(CPP) -c $(CFLAGS) $(CPPFILES)
	nvcc   -c $(NVCCFLAGS) $(CUDA_LIBS) *.cu
	$(CPP) -shared *.o $(CUDA_LIBS) -o core.so

test:	all  *_test.cpp
	$(CPP) $(CFLAGS) $(LDFLAGS) tensor_test.cpp tensor.o -o tensor_test
	$(CPP) $(CFLAGS) $(LDFLAGS) conv_test.cpp tensor.o -o conv_test
	./tensor_test
	./conv_test

.PHONY:	clean
clean:	
	rm -f *.o *.so iotest tensor_test conv_test
