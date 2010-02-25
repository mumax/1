export LD_CUDA_LIBS=-lcuda -lcudart -lcufft

all:	*.cu
	nvcc test.cu $(LD_CUDA_LIBS) -o test

test:	*_test.cu
	nvcc fft1D_test.cu $(LD_CUDA_LIBS) -o fft1D_test
	./fft1D_test
	nvcc fft3D_test.cu $(LD_CUDA_LIBS) -o fft3D_test
	./fft3D_test

.PHONY:	clean
clean:
	rm -f test *_test


