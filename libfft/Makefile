all:
	make -C libfft-cpu
	ln -sf libfft-cpu/libfft-cpu.so libfft.so

clean:
	make clean -C libfft-cpu
	rm -f *.so fft_test

test:	all
	g++ -Wall fft_test.cpp -o fft_test
	./fft_test
