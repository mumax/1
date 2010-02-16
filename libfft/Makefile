all:
	make -C libfft-cpu
	ln -s libfft-cpu/libfft-cpu.so libfft.so

clean:
	make clean -C libfft-cpu
	rm -f *.so
