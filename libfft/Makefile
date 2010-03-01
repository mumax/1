include ../common.mk

CLEANFILES+=../lib/libfft*.*

all:
	make -C libfft-cpu
	make -C libfft-cuda
	# Here we choose to use libfft-XXX as the default, can be changed.
	ln -sf ../libfft/libfft-cuda/libfft-cuda.so ../lib/libfft.so
	#ln -sf ../libfft/libfft-cuda/libfft-cpu.a ../lib/libfft.a

.PHONY: clean
clean:
	make clean -C libfft-cpu
	make clean -C libfft-cuda
	rm -f $(CLEANFILES) fft_test

test:	all
	$(CPP) $(CFLAGS) ../lib/libtensor.so ../lib/libfft.so $(LD_FFTW_LIBS) fft_test.cpp -o fft_test
	./fft_test
