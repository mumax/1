all: libfft-cpu.so libfft-cpu.a

libfft-cpu.o:	../libfft.h libfft_cpu.c
	gcc -Wall -fPIC -c -lfftw3f -lfftw3f_threads -lpthread libfft_cpu.c -o libfft-cpu.o

libfft-cpu.so:	libfft-cpu.o	
	gcc -Wall -shared -lfftw3f -lfftw3f_threads -lpthread libfft-cpu.o -o libfft-cpu.so

libfft-cpu.a:	libfft-cpu.o
	ar rcs libfft-cpu.a libfft-cpu.o

.PHONY: clean
clean:
	rm -f *.o *.so *.a
