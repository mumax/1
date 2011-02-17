include ../../common.mk

OFILES=\
	gpu_kernel_micromag3d.o\
	gpu_kernel_micromag2d.o\
	mumaxkern-gpu.o\

CFLAGS+=-I../gpukern -I../tensor -L../../lib

NVCCFLAGS+=-I../gpukern -I../tensor

all: mumaxkern-gpu

mumaxkern-gpu: $(OFILES)
	$(CPP) $(CFLAGS) libgpukern.so $(OFILES) -o mumaxkern-gpu

gpu_kernel_micromag2d.o: gpu_kernel_micromag2d.cu  gpu_kernel_micromag2d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag2d.cu

gpu_kernel_micromag3d.o: gpu_kernel_micromag3d.cu  gpu_kernel_micromag3d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag3d.cu

mumaxkern-gpu.o: mumaxkern-gpu.cpp mumaxkern-gpu.h
	$(CPP) $(CFLAGS) -c mumaxkern-gpu.cpp
	
.PHONY:	clean
clean:	
	rm -f *.o mumaxkern-gpu
