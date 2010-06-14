include ../common.mk

CPPFILES=\
	tensor.cpp\
	pipes.cpp\
	timer.cpp\
	units.c\

CUFILES=\
	gputil.cu\
	gpuconv1.cu\
	gpuconv2.cu\
	gpufft.cu\
	gpufft2.cu\
	gpueuler.cu\
	gpuheun.cu\
	gpurk4.cu\
	gpuanal1.cu\
	gpukernel1.cu\
	gpuheun2.cu\
# NOTE: commented files must be LAST in the list, otherwise all the rest is also commented out

OFILES=\
	tensor.o\
	pipes.o\
	timer.o\
	units.o\
	gputil.o\
	gpuconv1.o\
	gpuconv2.o\
	gpufft.o\
	gpufft2.o\
	gpueuler.o\
	gpuheun.o\
	gpurk4.o\
	gpuanal1.o\
	gpukernel1.o\
	gpuheun2.o\

#remove the "emu" when not using device emulation
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

gpuconv2.o:	gpuconv2.h gpuconv2.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuconv2.cu

gpufft.o:	gpufft.h gpufft.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpufft.cu

gpufft2.o:	gpufft2.h gpufft2.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpufft2.cu

gpueuler.o:	gpueuler.h gpueuler.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpueuler.cu

gpuheun.o:	gpuheun.h gpuheun.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuheun.cu

gpuheun2.o:	gpuheun2.h gpuheun2.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuheun2.cu

gpurk4.o:	gpurk4.h gpurk4.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpurk4.cu

gpuanal1.o:	gpuanal1.h gpuanal1.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuanal1.cu

gpukernel1.o:	gpukernel1.h gpukernel1.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpukernel1.cu

timer.o:	timer.h timer.cpp
	$(NVCC) -c $(NVCCFLAGS) timer.cpp

units.o:	units.h units.c
	$(NVCC) -c $(NVCCFLAGS) units.c

test:	$(OFILES) tensor_test gpueuler_test gpuheun_test gpuheun2_test gpuanal1_test gpusim_test gputil_test gpuconv2_test gpufft_test gpukernel1_test
	./tensor_test
	./gputil_test
	./gpufft_test
	./gpuconv2_test
	./gpueuler_test testm0.t testkernel.t
	./gpuheun_test testm0.t testkernel.t
	./gpuheun2_test
	./gpuanal1_test testm0.t testkernel.t
	./gpukernel1_test

# We link against core.so instead of $OFILES so we can use gcc instead of nvcc
tensor_test:	core.so tensor_test.cpp
	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
	$(CPP) tensor_test.o core.so -o tensor_test

gpueuler_test:	$(OFILES) gpueuler_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpueuler_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpueuler_test.o -o gpueuler_test

gpuheun_test:	$(OFILES) gpuheun_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuheun_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuheun_test.o -o gpuheun_test

gpuheun2_test:	$(OFILES) gpuheun2_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuheun2_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuheun2_test.o -o gpuheun2_test

gpuanal1_test:	$(OFILES) gpuanal1_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuanal1_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuanal1_test.o -o gpuanal1_test


gpusim_test:	$(OFILES) gpusim_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpusim_test.o -o gpusim_test
	ln -sf $(CURDIR)/gpusim_test ../../../bin/gpusim_test

gpuconv2_test:	$(OFILES) gpuconv2_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuconv2_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpuconv2_test.o -o gpuconv2_test

gpufft_test:	$(OFILES) gpufft_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpufft_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpufft_test.o -o gpufft_test

gpufft2_test:	$(OFILES) gpufft2_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpufft2_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpufft2_test.o -o gpufft2_test

gputil_test: $(OFILES) gputil_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gputil_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gputil_test.o -o gputil_test

gpukernel1_test: $(OFILES) gpukernel1_test.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpukernel1_test.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) gpukernel1_test.o -o gpukernel1_test


.PHONY:	clean
clean:	
	rm -f *.o *.so *.a iotest *_test

doc:
	doxygen doxygen.cfg