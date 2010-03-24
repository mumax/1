include ../../common.mk

CPPFILES=\
	main.cpp\

OFILES=\
	main.o\

CUDALIBS=-lcuda -lcudart -lcufft

all: main

main: $(OFILES)
	$(CPP) $(OFILES) ../../core/core.so  -o main

$(OFILES): *.h $(CPPFILES)
	$(CPP) -c $(CFLAGS) $(LDFLAGS) $(CPPFILES)

.PHONY:	clean
clean:	
	rm -f *.o main
