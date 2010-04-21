include ../../common.mk

CPPFILES=\
	main.cpp

OFILES=\
	main.o\

CUDALIBS= -lcudart -lcufft

all:	main.cpp
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) -I ../../core main.cpp
	$(NVCC) $(CUDALIBS) $(OFILES) main.o -o main

.PHONY:	clean
clean:	
	rm -f *.o main
