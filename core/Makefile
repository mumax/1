include ../common.mk

CPPFILES=\
	tensor.cpp\

CUFILES=\
	gputil.cu\
	gpuconv1.cu\
	gpueuler.cu\

OFILES=\
	tensor.o\
	gputil.o\
	gpuconv1.o\
	gpueuler.o\

CUDALIBS=-lcuda -lcudart -lcufft

all: core.so

core.so: $(OFILES)
	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so

$(OFILES): *.h $(CPPFILES) $(CUFILES)
	$(NVCC) -c $(NVCCFLAGS)  $(CPPFILES)
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) *.cu


test:	all  *_test.cpp
	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
	$(CPP) tensor_test.o core.so -o tensor_test
	./tensor_test
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpusim_test.o -o gpusim_test
# #	./gpusim_test m0.t kern.t h.t

.PHONY:	clean
clean:	
	rm -f *.o *.so iotest *_test

doc:	*
	doxygen doxygen.cfg


