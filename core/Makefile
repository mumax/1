include ../common.mk

all:	*.h *.cpp
	$(CPP) -c $(CFLAGS) *.cpp

test:	all
	$(CPP) $(LDFLAGS) tensor_test.o tensor.o -o tensor_test
	./tensor_test

.PHONY:	clean
clean:	
	rm -f *.o iotest tensor_test
