all: libtensor.so libtensor.a tensor

libtensor.o:	libtensor.h libtensor.cpp
	g++ -c -Wall -fPIC libtensor.cpp -o libtensor.o

libtensor.so:	libtensor.o
	g++ -Wall -shared libtensor.o -o libtensor.so

libtensor.a:	libtensor.o
	ar rcs libtensor.a libtensor.o

tensor:	main.cpp libtensor.a
	g++ -Wall main.cpp -L. -o tensor

test: tensor_test
	./tensor_test

tensor_test:	tensor_test.c libtensor.a
	g++ -c -Wall tensor_test.c -o tensor_test.o
	g++ -Wall -L. tensor_test.o -static libtensor.a -o tensor_test

.PHONY: clean
clean:
	rm -f *.o *.so *.a tensor_test tensor
