all:	*.cu
	nvcc test.cu -lcuda -lcudart -lcufft -o test

test:	fft1D_test.cu
	nvcc fft1D_test.cu -lcuda -lcudart -lcufft -o fft1D_test
	./fft1D_test

.PHONY:	clean
clean:
	rm -f test fft1D_test


