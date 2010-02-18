all: libtensor.so

libtensor.so:	libtensor.h libtensor.cpp
	g++ -fPIC -shared libtensor.cpp -o libtensor.so

test: tensor_test
	./tensor_test

tensor_test:	tensor_test.c libtensor.so
	g++ -c tensor_test.c -o tensor_test.o
	g++ -L. tensor_test.o libtensor.so -o tensor_test

.PHONY: clean
clean:
	rm -f *.o *.so tensor_test
