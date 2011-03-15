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
	gpu_transpose2.o\
	gpu_kernmul.o\
	gpu_zeropad.o\
	gpu_init.o\
	timer.o\
	gpu_stream.o\
	gpu_reduction.o\
	gpu_fftbig.o\
	gpu_fft0.o\
 	gpu_anal.o\
	gpu_spintorque.o\
	gpu_anis.o\
	gpu_local_contr2.o\
	gpu_exch1.o\
	gpu_temperature.o\

all: libgpukern.so

libgpukern.so: $(OFILES)
	$(NVCC) -shared $(NVCCFLAGS) $(OFILES) $(CUDALIBS) $(L) -o libgpukern.so
	cp -f $(CURDIR)/libgpukern.so $(CURDIR)/../../lib

libgpukern.a:	$(OFILES)
	ar rcs libgpukern.a $(OFILES)
# 	ln -sf $(CURDIR)/libgpukern.a $(CURDIR)/../../lib

timer.o:  timer.h timer.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) timer.cpp

gpu_stream.o:  gpu_stream.h gpu_stream.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_stream.cu

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

gpu_transpose2.o:  gpu_transpose2.h gpu_transpose2.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_transpose2.cu

# gpu_fft4.o:  gpu_fft4.h gpu_fft4.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft4.cu
# 
# gpu_fft3.o:  gpu_fft3.h gpu_fft3.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft3.cu

gpu_fft0.o:  gpu_fft0.h gpu_fft0.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft0.cu

gpu_kernmul.o:  gpu_kernmul.h gpu_kernmul.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_kernmul.cu

gpu_zeropad.o:  gpu_zeropad.h gpu_zeropad.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_zeropad.cu

gpu_init.o:  gpu_init.h gpu_init.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_init.cu

gpu_reduction.o:  gpu_reduction.h gpu_reduction.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_reduction.cu

# gpu_bigfft.o:  gpu_bigfft.h gpu_bigfft.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_bigfft.cu

gpu_fftbig.o: gpu_fftbig.h gpu_fftbig.cu
	$(NVCC)  -c $(CUDALIBS) $(NVCCFLAGS) gpu_fftbig.cu

gpu_fft6.o:  gpu_fft6.h gpu_fft6.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_fft6.cu

gpu_spintorque.o:  gpu_spintorque.h gpu_spintorque.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_spintorque.cu

gpu_anal.o:	gpu_anal.h gpu_anal.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_anal.cu

gpu_anis.o:  gpu_anis.h gpu_anis.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_anis.cu

gpu_local_contr2.o:  gpu_local_contr2.h gpu_local_contr2.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_local_contr2.cu

gpu_temperature.o:  gpu_temperature.h gpu_temperature.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_temperature.cu

# gpu_exch0.o:  gpu_exch.h gpu_exch0.cu
# 	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_exch0.cu

gpu_exch1.o:  gpu_exch.h gpu_exch1.cu
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) gpu_exch1.cu

test:	all transpose_test zeropad_test reduction_test fft_test spintorque_test

spintorque_test:	spintorque_test.cpp libgpukern.so
	$(CPP) $(CFLAGS) libgpukern.so spintorque_test.cpp -o spintorque_test
	./spintorque_test

transpose_test:	transpose_test.cpp libgpukern.so
	$(CPP) $(CFLAGS) libgpukern.so transpose_test.cpp -o transpose_test
	./transpose_test

zeropad_test:	zeropad_test.cpp libgpukern.so
	$(CPP) $(CFLAGS) libgpukern.so zeropad_test.cpp -o zeropad_test
	./zeropad_test

reduction_test:	reduction_test.cpp libgpukern.so
	$(CPP) -g libgpukern.so reduction_test.cpp -o reduction_test
	./reduction_test

fft_test:	fft_test.cpp libgpukern.so
	$(CPP) libgpukern.so fft_test.cpp -o fft_test
	./fft_test

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a
