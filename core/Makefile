include ../common.mk

CPPFILES=\
	tensor.cpp\

CUFILES=\
	gputil.cu\
	gpuconv1.cu\
	gpueuler.cu\
	gpurk4.cu\

OFILES=\
	tensor.o\
	gputil.o\
	gpuconv1.o\
	gpueuler.o\
	gpurk4.o\

CUDALIBS= -lcudart -lcufft

all: core.so

core.so: $(OFILES)
	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so

$(OFILES): *.h $(CPPFILES) $(CUFILES)
	$(NVCC) -c $(NVCCFLAGS)  $(CPPFILES)
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) $(CUFILES)


test:	all  *_test.cpp
	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
	$(CPP) tensor_test.o core.so -o tensor_test
	./tensor_test
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpusim_test.o -o gpusim_test
	ln -sf $(CURDIR)/gpusim_test ../../../bin/gpusim_test

.PHONY:	clean
clean:	
	rm -f *.o *.so iotest *_test

doc:	*
	doxygen doxygen.cfg


