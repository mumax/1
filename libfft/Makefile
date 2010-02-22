all:
	make -C libfft-cpu
	# Here we choose to use libfft-cpu as the default, can be changed.
	ln -sf ../libfft/libfft-cpu/libfft-cpu.so ../lib/libfft.so
	ln -sf ../libfft/libfft-cpu/libfft-cpu.a ../lib/libfft.a


clean:
	make clean -C libfft-cpu
	rm -f *.so *.o *.a fft_test ../lib/libfft*.*

test:	all
	g++ ../lib/libtensor.so ../lib/libfft.so -lfftw3f fft_test.cpp -o fft_test
	./fft_test
