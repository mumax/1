all: libtensor.so

libtensor.o:	libtensor.h libtensor.c	
	gcc -fPIC -c -O3 libtensor.c -o libtensor.o

libtensor.so:	libtensor.o
	ld -shared libtensor.o -o libtensor.so

.PHONY: clean
clean:
	rm -f *.o *.so
