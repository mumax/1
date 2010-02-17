all: libtensor.so

libtensor.o:	libtensor.h libtensor.cpp	
	g++ -fPIC -c -O3 libtensor.cpp -o libtensor.o

libtensor.so:	libtensor.o
	ld -shared libtensor.o -o libtensor.so

.PHONY: clean
clean:
	rm -f *.o *.so
