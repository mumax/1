export LD_CUDA_LIBS=-lcuda -lcudart -lcufft

all:	libfft_cuda.cu
	nvcc --compiler-options -fPIC -c libfft_cuda.cu $(LD_CUDA_LIBS)
	nvcc -shared libfft_cuda.o $(LD_CUDA_LIBS) -o libfft-cuda.so

test:	*_test.cu
	nvcc fft1D_test.cu $(LD_CUDA_LIBS) -o fft1D_test
	./fft1D_test
	nvcc fft3D_test.cu $(LD_CUDA_LIBS) -o fft3D_test
	./fft3D_test

.PHONY:	clean
clean:
	rm -f test *_test *.o *.so


