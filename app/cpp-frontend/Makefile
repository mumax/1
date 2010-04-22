include ../../common.mk

CPPFILES=\
	main.cpp

OFILES=\
	main.o\

CUDALIBS= -lcudart -lcufft

all:	main.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) -I ../../core main.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) ../../core/core.a -o main
	ln -sf $(CURDIR)/main ../../bin/cppsim 

.PHONY:	clean
clean:	
	rm -f *.o main ../../bin/cppsim
