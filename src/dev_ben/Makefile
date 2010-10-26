include common.mk

CPPFILES=\
	tensor.cpp\
	timer.cpp\
	param.cpp\
	field.cpp\
	timestep.cpp\
	cpu_exch.cpp\
	main-ben.cpp\
	kernel.cpp\
#	pipes.cpp\
#	debug.cpp\


OFILES=\
	tensor.o\
	param.o\
	timer.o\
	gputil.o\
	gpu_anal.o\
	field.o\
	timestep.o\
	kernel.o\
	gpueuler.o\
	gpu_micromag3d_kernel.o\
	gpu_micromag2d_kernel.o\
	gpu_conv.o\
	cpu_exch.o\
	gpu_exchange.o\
	gpu_heun.o\
# 	gpu_fftbig.o\
#	gpu_fft6.o\
# 	gpu_fft4.o\
# 	gpu_fft5.o
# 	gpu_transpose2.o\
# 	gpunormalize.o\
#	gputorque.o\
# 	gpufft2.o\
# 	gpu_conf.o\
# 	gpurk4.o\
#	gpukernel1.o\
#	debug.o\
#	pipes.o\
#	gpupad.o\
#	gpuconv1.o\
#	gpuconv2.o\

# all: core.so core.a doc

tensor.o:	tensor.h tensor.cpp
	$(NVCC) -c $(NVCCFLAGS) tensor.cpp

param.o:	param.h param.cpp
	$(NVCC) -c $(NVCCFLAGS) param.cpp

gputil.o:	gputil.h gputil.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gputil.cu

gpueuler.o:	gpueuler.h gpueuler.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpueuler.cu

gpu_heun.o:	gpu_heun.h gpu_heun.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_heun.cu

# gpunormalize.o:	gpunormalize.h gpunormalize.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpunormalize.cu

# gputorque.o:	gputorque.h gputorque.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gputorque.cu

gpu_micromag3d_kernel.o:	gpu_micromag3d_kernel.h gpu_micromag3d_kernel.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_micromag3d_kernel.cu

gpu_micromag2d_kernel.o:	gpu_micromag2d_kernel.h gpu_micromag2d_kernel.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_micromag2d_kernel.cu

gpu_anal.o:	gpu_anal.h gpu_anal.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_anal.cu

gpu_conv.o:	gpu_conv.h gpu_conv.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_conv.cu

timer.o:	timer.h timer.cpp
	$(NVCC) -c $(NVCCFLAGS) timer.cpp

field.o:	field.h field.cpp
	$(NVCC) -c $(NVCCFLAGS) field.cpp

timestep.o:	timestep.h timestep.cpp
	$(NVCC) -c $(NVCCFLAGS) timestep.cpp

kernel.o:	kernel.h kernel.cpp
	$(NVCC) -c $(NVCCFLAGS) kernel.cpp

cpu_exch.o:	cpu_exch.h cpu_exch.cpp
	$(NVCC) -c $(NVCCFLAGS) cpu_exch.cpp

gpu_exchange.o:	gpu_exchange.h gpu_exchange.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_exchange.cu

# gpu_transpose2.o: gpu_transpose2.h gpu_transpose2.cu
# 	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_transpose2.cu

# gpu_fftbig.o: gpu_fftbig.h gpu_fftbig.cu
# 	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_fftbig.cu

# gpu_fft4.o: gpu_fft4.h gpu_fft4.cu
# 	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft4.cu
# 
# gpu_fft5.o: gpu_fft5.h gpu_fft5.cu
# 	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft5.cu

# gpu_fft6.o: gpu_fft6.h gpu_fft6.cu
# 	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft6.cu

main-ben.o:	main-ben.h main-ben.cpp
	$(NVCC) -c $(NVCCFLAGS) main-ben.cpp


main-ben: $(OFILES) main-ben.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) main-ben.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) ../gpukern/libgpukern.so main-ben.o -o main-ben




#core.so: $(OFILES)
#	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so
#
#core.a:	$(OFILES)
#	ar rcs core.a $(OFILES)
#
#debug.o:	debug.h debug.cpp
#	$(CPP) -c $(CFLAGS) debug.cpp
#
#pipes.o:	pipes.h pipes.cpp
#	$(NVCC) -c $(NVCCFLAGS) pipes.cpp
#
#gpupad.o:	gpupad.h gpupad.cu
#	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpupad.cu
#
#gpuconv1.o:	gpuconv1.h gpuconv1.cu
#	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuconv1.cu
#
#gpuconv2.o:	gpuconv2.h gpuconv2.cu
#	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpuconv2.cu
#
#gpufft.o:	gpufft.h gpufft.cu
#	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpufft.cu
#
#gpukernel1.o:	gpukernel1.h gpukernel1.cu
#	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpukernel1.cu
# 
# gpurk4.o:	gpurk4.h gpurk4.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpurk4.cu
# 
# gpufft2.o:	gpufft2.h gpufft2.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpufft2.cu



#test:	$(OFILES) tensor_test gpusim_test gputil_test gpuconv2_test gpufft_test gpufft2_test
#	./tensor_test
#	./gputil_test
#	./gpueuler_test testm0.t testkernel.t
#	./gpuheun_test testm0.t testkernel.t
#	./gpuheun2_test
#	./gpufft_test
#	./gpufft2_test
#	./gpuconv2_test
#	./gpukernel1_test

# We link against core.so instead of $OFILES so we can use gcc instead of nvcc
#tensor_test:	core.so tensor_test.cpp
#	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
#	$(CPP) tensor_test.o core.so -o tensor_test
#
#gpueuler_test:	$(OFILES) gpueuler_test.cpp
#	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpueuler_test.cpp
#	$(NVCC) $(CUDALIBS) $(OFILES) gpueuler_test.o -o gpueuler_test
#
#gpuheun_test:	$(OFILES) gpuheun_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuheun_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpuheun_test.o -o gpuheun_test
# 
# 
# gpusim_test:	$(OFILES) gpusim_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpusim_test.o -o gpusim_test
# #	ln -sf $(CURDIR)/gpusim_test ../../../bin/gpusim_test
# 
# gpuconv2_test:	$(OFILES) gpuconv2_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpuconv2_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpuconv2_test.o -o gpuconv2_test
# 
# gpufft_test:	$(OFILES) gpufft_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpufft_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpufft_test.o -o gpufft_test
# 
# gpufft2_test:	$(OFILES) gpufft2_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpufft2_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpufft2_test.o -o gpufft2_test
# 
# gputil_test: $(OFILES) gputil_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gputil_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gputil_test.o -o gputil_test
# 
# gpu_micromag3d_kernel_test: $(OFILES) gpu_micromag3d_kernel_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpu_micromag3d_kernel_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpu_micromag3d_kernel_test.o -o gpu_micromag3d_kernel_test
# 
# gpu_micromag2d_kernel_test: $(OFILES) gpu_micromag2d_kernel_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpu_micromag2d_kernel_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpu_micromag2d_kernel_test.o -o gpu_micromag2d_kernel_test
# 
# gpukernel1_test: $(OFILES) gpukernel1_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpukernel1_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpukernel1_test.o -o gpukernel1_test
# 
# gpu_conv_test: $(OFILES) gpu_conv_test.cpp
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpu_conv_test.cpp
# 	$(NVCC) $(CUDALIBS) $(OFILES) gpu_conv_test.o -o gpu_conv_test



.PHONY:	clean
clean:	
	rm -f *.o *.so *.a iotest *_test

doc:
	doxygen doxygen.cfg