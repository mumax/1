all: libtensor.so libtensor.a

libtensor.o:	libtensor.h libtensor.cpp
	g++ -c -fPIC libtensor.cpp -o libtensor.o

libtensor.so:	libtensor.o
	g++ -shared libtensor.o -o libtensor.so

libtensor.a:	libtensor.o
	ar rcs libtensor.a libtensor.o

test: tensor_test
	./tensor_test

tensor_test:	tensor_test.c libtensor.so
	g++ -c tensor_test.c -o tensor_test.o
	g++ -L. tensor_test.o libtensor.so -o tensor_test

.PHONY: clean
clean:
	rm -f *.o *.so *.a tensor_test
