all:
	make -C libfft-cpu
	# Here we choose to use libfft-cpu as the default, can be changed.
	ln -sf libfft-cpu/libfft-cpu.so ../lib/libfft.so
	ln -sf libfft-cpu/libfft-cpu.a ../lib/libfft.a


clean:
	make clean -C libfft-cpu
	rm -f *.so *.o *.a fft_test ../lib/libfft*.*

test:	all
	g++ -Wall fft_test.cpp -o fft_test
	./fft_test
