include ../../common.mk


main:	main.cpp
	$(CPP) -c $(CFLAGS) -I ../../core main.cpp
	$(CPP) $(LDFLAGS) main.o ../../core/core.so -o main

main-arne:	main-arne.cpp
	$(CPP) -c $(CFLAGS) -I ../../core main-arne.cpp
	$(CPP) $(LDFLAGS) main-arne.o ../../core/core.so -o main-arne


.PHONY:	clean
clean:	
	rm -f *.o main main-arne
