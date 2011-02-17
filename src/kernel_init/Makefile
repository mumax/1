include ../../common.mk

CFLAGS+=-I../gpukern -I../tensor -L../../lib

NVCCFLAGS+=-I../gpukern -I../tensor

all: mumaxkern-gpu mumaxkern-cpu

mumaxkern-gpu: gpu_kernel_micromag2d.o gpu_kernel_micromag3d.o mumaxkern-gpu.o
	$(CPP) $(CFLAGS) -lgpukern $(OFILES) ../tensor/tensor.o -o mumaxkern-gpu
	cp -f mumaxkern-gpu ../../bin

mumaxkern-cpu: gpu_kernel_micromag2d.o gpu_kernel_micromag3d.o mumaxkern-cpu.o
	$(CPP) $(CFLAGS) -lgpukern $(OFILES) ../tensor/tensor.o -o mumaxkern-cpu
	cp -f mumaxkern-cpu ../../bin


gpu_kernel_micromag2d.o: gpu_kernel_micromag2d.cu  gpu_kernel_micromag2d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag2d.cu

cpu_kernel_micromag2d.o: cpu_kernel_micromag2d.cu  cpu_kernel_micromag2d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) cpu_kernel_micromag2d.cu


gpu_kernel_micromag3d.o: gpu_kernel_micromag3d.cu  gpu_kernel_micromag3d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag3d.cu

gpu_kernel_micromag3d.o: gpu_kernel_micromag3d.cu  gpu_kernel_micromag3d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag3d.cu

mumaxkern-gpu.o: mumaxkern-gpu.cpp mumaxkern-gpu.h
	$(CPP) $(CFLAGS) -c mumaxkern-gpu.cpp
	
.PHONY:	clean
clean:	
	rm -f *.o mumaxkern-gpu
