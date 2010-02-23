all: libfft-cpu.so libfft-cpu.a

libfft-cpu.o:	../libfft.h libfft_cpu.c
	$(CC) $(CFLAGS) -c $(LD_FFTW_LIBS) libfft_cpu.c -o libfft-cpu.o

libfft-cpu.so:	libfft-cpu.o	
	$(CC) $(CFLAGS) -shared $(LD_FFTW_LIBS) libfft-cpu.o -o libfft-cpu.so

libfft-cpu.a:	libfft-cpu.o
	ar rcs libfft-cpu.a libfft-cpu.o

.PHONY: clean
clean:
	rm -f *.o *.so *.a
