include ../common.mk

CPPFILES=\
	tensor.cpp\

CUFILES=\
	*.cu\

OFILES=\
	tensor.o\
	conv_gpu.o\

# CUDALIBS=-L /usr/local/cuda/lib/libcudart.so -L /usr/local/cuda/lib/libcufft.so


all: core.so

core.so: $(OFILES)
	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so

$(OFILES): *.h $(CPPFILES) $(CUFILES)
	$(CPP)  -c $(CFLAGS) $(CPPFILES)
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_LIBS) *.cu


test:	all  *_test.cpp
	$(CPP) $(CFLAGS) $(LDFLAGS) tensor_test.cpp core.so -o tensor_test
	$(CPP) $(CFLAGS) $(LDFLAGS) conv_test.cpp core.so -o conv_test
	./tensor_test
	./conv_test


.PHONY:	clean
clean:	
	rm -f *.o *.so iotest *_test
