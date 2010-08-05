include ../../common.mk

OFILES=\
	cpu_properties.o\
	cpu_mem.o\
	cpu_linalg.o\
	cpu_normalize.o\
	cpu_torque.o\
	cpu_anal.o\
	cpu_kernmul.o\
	cpu_zeropad.o\
	cpu_transpose.o\

all: libcpukern.so libcpukern.a

libcpukern.so: $(OFILES)
	$(CPP) -shared $(CFLAGS) $(OFILES) -o libcpukern.so

libcpukern.a:	$(OFILES)
	ar rcs libcpukern.a $(OFILES)


cpu_properties.o:  cpu_properties.h cpu_properties.cpp                                                
	$(CPP) -c $(CFLAGS) cpu_properties.cpp

cpu_mem.o:  cpu_mem.h cpu_mem.cpp
	$(CPP) -c $(CFLAGS) cpu_mem.cpp

cpu_linalg.o:  cpu_linalg.h cpu_linalg.cpp
	$(CPP) -c $(CFLAGS) cpu_linalg.cpp

cpu_normalize.o:  cpu_normalize.h cpu_normalize.cpp
	$(CPP) -c $(CFLAGS) cpu_normalize.cpp

cpu_torque.o:  cpu_torque.h cpu_torque.cpp
	$(CPP) -c $(CFLAGS) cpu_torque.cpp

cpu_anal.o:  cpu_anal.h cpu_anal.cpp
	$(CPP) -c $(CFLAGS) cpu_anal.cpp

cpu_kernmul.o:  cpu_kernmul.h cpu_kernmul.cpp
	$(CPP) -c $(CFLAGS) cpu_kernmul.cpp

cpu_zeropad.o:  cpu_zeropad.h cpu_zeropad.cpp
	$(CPP) -c $(CFLAGS) cpu_zeropad.cpp

cpu_transpose.o:  cpu_transpose.h cpu_transpose.cpp
	$(CPP) -c $(CFLAGS) cpu_transpose.cpp

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a
