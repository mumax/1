include ../../common.mk

OFILES=\
	gpu_safe.o\
	gpu_properties.o\
	gpu_conf.o\
	gpu_mem.o\
	gpu_linalg.o\
	gpu_normalize.o\
	gpu_torque.o\
	gpu_transpose.o\
	gpu_fft.o\
	gpu_kernmul.o\
	gpu_zeropad.o\
	gpu_anal.o\
	gpu_init.o\

all: libgpukern.so libgpukern.a

libgpukern.so: $(OFILES)
	$(NVCC) -shared $(NVCCFLAGS) $(OFILES) $(CUDALIBS) $(L) -o libgpukern.so
	ln -sf $(CURDIR)/libgpukern.so $(CURDIR)/../../lib

libgpukern.a:	$(OFILES)
	ar rcs libgpukern.a $(OFILES)
	ln -sf $(CURDIR)/libgpukern.a $(CURDIR)/../../lib

gpu_safe.o:  gpu_safe.h gpu_safe.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_safe.cu

gpu_properties.o:  gpu_properties.h gpu_properties.cu                                                
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_properties.cu 

gpu_conf.o:  gpu_conf.h gpu_conf.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_conf.cu

gpu_mem.o:  gpu_mem.h gpu_mem.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_mem.cu

gpu_linalg.o:  gpu_linalg.h gpu_linalg.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_linalg.cu

gpu_normalize.o:  gpu_normalize.h gpu_normalize.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_normalize.cu

gpu_torque.o:  gpu_torque.h gpu_torque.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_torque.cu

gpu_transpose.o:  gpu_transpose.h gpu_transpose.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_transpose.cu

gpu_fft.o:  gpu_fft.h gpu_fft.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft.cu

gpu_kernmul.o:  gpu_kernmul.h gpu_kernmul.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernmul.cu

gpu_zeropad.o:  gpu_zeropad.h gpu_zeropad.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_zeropad.cu

gpu_anal.o:  gpu_anal.h gpu_anal.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_anal.cu

gpu_init.o:  gpu_init.h gpu_init.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_init.cu

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a
