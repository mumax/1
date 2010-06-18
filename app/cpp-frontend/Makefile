include ../../common.mk

CPPFILES=\
	main.cpp

OFILES=\
	main.o\


all:	main.cpp
	$(CPP) -c $(CFLAGS) -I ../../core main.cpp
	$(CPP) $(LDFLAGS) $(OFILES) ../../core/core.so -o main
	ln -sf $(CURDIR)/main ../../bin/cppsim 

.PHONY:	clean
clean:	
	rm -f *.o main ../../bin/cppsim
