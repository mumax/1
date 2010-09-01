include ../../common.mk

OFILES=\
	tensor.o\

all: libtensor.so libtensor.a

libtensor.so: $(OFILES)
	$(CPP) -shared $(CFLAGS)  $(OFILES) -Wl,-soname,libtensor.so -o libtensor.so
	ln -sf $(CURDIR)/libtensor.so $(CURDIR)/../../lib/libtensor.so

libtensor.a: $(OFILES)
	ar rcs libtensor.a $(OFILES) $(FFTWLIBS)
	ln -sf $(CURDIR)/libtensor.a $(CURDIR)/../../lib

tensor.o:  tensor.h tensor.cpp
	$(CPP) -c $(CFLAGS)  tensor.cpp

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a
