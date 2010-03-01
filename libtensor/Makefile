include ../common.mk

all: libtensor.so libtensor.a tensor install

libtensor.o:	libtensor.h libtensor.cpp
	$(CPP) $(CFLAGS) -c libtensor.cpp -o libtensor.o

libtensor.so:	libtensor.o
	$(CPP) $(CFLAGS) -shared libtensor.o -o libtensor.so

libtensor.a:	libtensor.o
	ar rcs libtensor.a libtensor.o

tensor:	main.cpp libtensor.a
	$(CPP) $(CFLAGS) main.cpp -L. -o tensor

test: tensor_test
	./tensor_test

tensor_test:	tensor_test.c libtensor.a
	$(CPP) $(CFLAGS) -c  tensor_test.c -o tensor_test.o
	$(CPP) $(CFLAGS) -L. tensor_test.o -static libtensor.a -o tensor_test

install: libtensor.so libtensor.a
	ln -sf ../libtensor/libtensor.so ../lib/libtensor.so
	ln -sf ../libtensor/libtensor.a ../lib/libtensor.a

.PHONY: clean
clean:
	rm -f $(CLEANFILES) tensor_test tensor ../lib/libtensor.*
