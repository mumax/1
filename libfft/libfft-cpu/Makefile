all: libfft-cpu.so

libfft-cpu.o:	../libfft.h libfft_cpu.c	
	gcc -fPIC -c -lfftw3 -O3 libfft_cpu.c -o libfft-cpu.o

libfft-cpu.so:	libfft-cpu.o
	ld -shared libfft-cpu.o -o libfft-cpu.so

.PHONY: clean
clean:
	rm -f *.o *.so
