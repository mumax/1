include ../common.mk

CPPFILES=\
	tensor.cpp\
	pipes.cpp\
	timer.cpp\
	units.c\

CUFILES=\
	gputil.cu\
	gpuconv1.cu\
	gpueuler.cu\
	gpuheun.cu\
	gpurk4.cu\
	gpuanal1.cu\

OFILES=\
	tensor.o\
	pipes.o\
	timer.o\
	units.o\
	gputil.o\
	gpuconv1.o\
	gpueuler.o\
	gpuheun.o\
	gpurk4.o\
	gpuanal1.o\

CUDALIBS= -lcudart -lcufft

all: core.so core.a doc

core.so: $(OFILES)
	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so

core.a:	$(OFILES)
	ar rcs core.a $(OFILES)

tensor.o:	tensor.h tensor.cpp
	$(NVCC) -c $(NVCCFLAGS) tensor.cpp

pipes.o:	pipes.h pipes.cpp
	$(NVCC) -c $(NVCCFLAGS) pipes.cpp

gputil.o:	gputil.h gputil.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gputil.cu

gpuconv1.o:	gpuconv1.h gpuconv1.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuconv1.cu

gpueuler.o:	gpueuler.h gpueuler.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpueuler.cu

gpuheun.o:	gpuheun.h gpuheun.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuheun.cu

gpurk4.o:	gpurk4.h gpurk4.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpurk4.cu

gpuanal1.o:	gpuanal1.h gpuanal1.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuanal1.cu

timer.o:	timer.h timer.cpp
	$(NVCC) -c $(NVCCFLAGS) timer.cpp

units.o:	units.h units.c
	$(NVCC) -c $(NVCCFLAGS) units.c


test:	all  $(OFILES) tensor_test gpueuler_test gpuheun_test gpuanal1_test gpusim_test

tensor_test:	core.so tensor_test.cpp
	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
	$(CPP) tensor_test.o core.so -o tensor_test
	./tensor_test

gpueuler_test:	all gpueuler_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpueuler_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpueuler_test.o -o gpueuler_test
	./gpueuler_test testm0.t testkernel.t

gpuheun_test:	all gpuheun_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuheun_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuheun_test.o -o gpuheun_test
	./gpuheun_test testm0.t testkernel.t

gpuanal1_test:	all gpuanal1_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuanal1_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuanal1_test.o -o gpuanal1_test
	./gpuanal1_test testm0.t testkernel.t

gpusim_test:	all gpusim_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpusim_test.o -o gpusim_test
	ln -sf $(CURDIR)/gpusim_test ../../../bin/gpusim_test

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a iotest *_test

doc:	*
	doxygen doxygen.cfg