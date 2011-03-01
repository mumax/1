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
	gpu_init_anal.o\
	gpu_mem.o\
	gpu_conf.o\
	gpu_safe.o\
# 	gpu_fftbig.o\
#	gpu_fft6.o\
# 	gpu_fft4.o\
# 	gpu_fft5.o

all: main

main.o:	main.h main.cpp
	$(NVCC) -c $(NVCCFLAGS) main.cpp

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

gpu_micromag3d_kernel.o:	gpu_micromag3d_kernel.h gpu_micromag3d_kernel.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_micromag3d_kernel.cu

gpu_micromag2d_kernel.o:	gpu_micromag2d_kernel.h gpu_micromag2d_kernel.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_micromag2d_kernel.cu

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

gpu_init_anal.o:	gpu_init_anal.h gpu_init_anal.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_init_anal.cu

gpu_mem.o:	gpu_mem.h gpu_mem.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_mem.cu

gpu_conf.o:	gpu_conf.h gpu_conf.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_conf.cu

gpu_safe.o:	gpu_safe.h gpu_safe.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_safe.cu

main: $(OFILES) main.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) main.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) ../gpukern/libgpukern.so main.o -o main


.PHONY:	clean
clean:	
	rm -f *.o *.so *.a iotest *_test

doc:
	doxygen doxygen.cfg