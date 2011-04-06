include ../../common.mk
FFTWLIBS=$(CURDIR)/../../lib/fftw/lib/libfftw3f_threads.a $(CURDIR)/../../lib/fftw/lib/libfftw3f.a
OMPFLAGS= -fopenmp
# OMPFLAGS= 
CFLAGS+=-I../../lib/fftw/include/

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
	cpu_init.o\
	cpu_reduction.o\
	cpu_fft2.o\
	cpu_local_contr.o\
	thread_functions.o\
	cpu_exch1.o

all: libcpukern.so

libcpukern.so: $(OFILES)
	$(CPP) -shared $(CFLAGS) $(OMPFLAGS) $(OFILES) $(FFTWLIBS) -Wl,-soname,libcpukern.so -o libcpukern.so
	cp -f $(CURDIR)/libcpukern.so $(CURDIR)/../../lib/libcpukern.so
	

libcpukern.a:	$(OFILES)
	ar rcs libcpukern.a $(OFILES) $(FFTWLIBS)
# 	ln -sf $(CURDIR)/libcpukern.a $(CURDIR)/../../lib


thread_functions.o:  thread_functions.h thread_functions.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) thread_functions.cpp

cpu_mem0.o:  cpu_mem.h cpu_mem0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_mem0.cpp

cpu_linalg0.o:  cpu_linalg.h cpu_linalg0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_linalg0.cpp

cpu_normalize0.o:  cpu_normalize.h cpu_normalize0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_normalize0.cpp

cpu_torque0.o:  cpu_torque.h cpu_torque0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_torque0.cpp

cpu_anal0.o:  cpu_anal.h cpu_anal0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_anal0.cpp

cpu_kernmul0.o:  cpu_kernmul.h cpu_kernmul0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_kernmul0.cpp

cpu_zeropad0.o:  cpu_zeropad.h cpu_zeropad0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_zeropad0.cpp

cpu_transpose0.o:  cpu_transpose.h cpu_transpose0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_transpose0.cpp

cpu_fft0.o:  cpu_fft.h cpu_fft0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_fft0.cpp

cpu_fft2.o:  cpu_fft.h cpu_fft2.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_fft2.cpp

cpu_fft1.o:  cpu_fft1.h cpu_fft1.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_fft1.cpp

cpu_init0.o:  cpu_init.h cpu_init0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_init0.cpp

cpu_reduction0.o:  cpu_reduction.h cpu_reduction0.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_reduction0.cpp

cpu_properties.o:  cpu_properties.h cpu_properties.cpp                                                
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_properties.cpp

cpu_mem.o:  cpu_mem.h cpu_mem.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_mem.cpp

cpu_linalg.o:  cpu_linalg.h cpu_linalg.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_linalg.cpp

cpu_normalize.o:  cpu_normalize.h cpu_normalize.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_normalize.cpp

cpu_torque.o:  cpu_torque.h cpu_torque.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_torque.cpp

cpu_anal.o:  cpu_anal.h cpu_anal.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_anal.cpp

cpu_kernmul.o:  cpu_kernmul.h cpu_kernmul.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_kernmul.cpp

cpu_zeropad.o:  cpu_zeropad.h cpu_zeropad.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_zeropad.cpp

cpu_transpose.o:  cpu_transpose.h cpu_transpose.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_transpose.cpp

cpu_fft.o:  cpu_fft.h cpu_fft.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_fft.cpp

# cpu_fft1.o:  cpu_fft1.h cpu_fft1.cpp
# 	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_fft1.cpp
# 
cpu_init.o:  cpu_init.h cpu_init.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_init.cpp

cpu_reduction.o:  cpu_reduction.h cpu_reduction.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_reduction.cpp

cpu_local_contr.o:  cpu_local_contr.h cpu_local_contr.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_local_contr.cpp

# thread_functions.o:  thread_functions.h thread_functions.cpp
# 	$(CPP) -c $(CFLAGS) $(OMPFLAGS) thread_functions.cpp
# 
cpu_exch1.o:  cpu_exch.h cpu_exch1.cpp
	$(CPP) -c $(CFLAGS) $(OMPFLAGS) cpu_exch1.cpp


test:

.PHONY:	clean
clean:	
	rm -f *.o *.so *.a
