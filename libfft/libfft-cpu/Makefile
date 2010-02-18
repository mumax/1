all: libfft-cpu.so

libfft-cpu.so:	../libfft.h libfft_cpu.c	
	gcc -fPIC -c -lfftw3 -O3 libfft_cpu.c -shared -o libfft-cpu.so

.PHONY: clean
clean:
	rm -f *.o *.so
