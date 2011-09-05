include ../../common.mk

CFLAGS+=-I../tensor -L../../lib

NVCCFLAGS+=-I../cpukern -I../gpukern -I../tensor -L../../lib

all: mumaxkern-gpu 


mumaxkern-gpu: gpu_kernel_micromag2d.o gpu_kernel_micromag3d.o mumaxkern-gpu.o
	$(CPP) $(CFLAGS) -lcpukern -lgpukern $(OFILES) ../tensor/tensor.o gpu_kernel_micromag2d.o gpu_kernel_micromag3d.o mumaxkern-gpu.o -o mumaxkern-gpu
	cp -f mumaxkern-gpu ../../bin

mumaxkern-cpu: cpu_kernel_micromag2d.o cpu_kernel_micromag3d.o mumaxkern-cpu.o
	$(CPP) $(CFLAGS) -lcpukern $(OFILES) ../tensor/tensor.o cpu_kernel_micromag2d.o cpu_kernel_micromag3d.o mumaxkern-cpu.o -o mumaxkern-cpu
	cp -f mumaxkern-cpu ../../bin


gpu_kernel_micromag2d.o: gpu_kernel_micromag2d.cu  gpu_kernel_micromag2d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag2d.cu

cpu_kernel_micromag2d.o: cpu_kernel_micromag2d.cpp  cpu_kernel_micromag2d.h
	$(CPP) -c $(CFLAGS) cpu_kernel_micromag2d.cpp


gpu_kernel_micromag3d.o: gpu_kernel_micromag3d.cu  gpu_kernel_micromag3d.h
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernel_micromag3d.cu

cpu_kernel_micromag3d.o: cpu_kernel_micromag3d.cpp cpu_kernel_micromag3d.h
	$(CPP) -c $(CFLAGS) cpu_kernel_micromag3d.cpp


mumaxkern-gpu.o: mumaxkern-gpu.cpp mumaxkern-gpu.h
	$(CPP) $(CFLAGS) -c mumaxkern-gpu.cpp
	
mumaxkern-cpu.o: mumaxkern-cpu.cpp mumaxkern-cpu.h
	$(CPP) $(CFLAGS) -c mumaxkern-cpu.cpp
	
.PHONY:	clean
clean:	
	rm -f *.o mumaxkern-gpu mumaxkern-cpu
