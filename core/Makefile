include ../common.mk

CPPFILES=\
	tensor.cpp\
	euler.cpp\

CUFILES=\
	*.cu\

OFILES=\
	tensor.o\
	conv_gpu.o\
	euler.o\
	gpusim.o\

CUDALIBS=-lcuda -lcudart -lcufft

all: core.so

core.so: $(OFILES)
	$(NVCC) -shared $(LDFLAGS) $(OFILES) $(CUDALIBS) -o core.so

# core.o: $(OFILES)
# 	ld -r $(OFILES) -o core.o

$(OFILES): *.h $(CPPFILES) $(CUFILES)
#	$(CPP)  -c $(CFLAGS) $(CPPFILES)
	$(NVCC) -c $(NVCCFLAGS)  $(CPPFILES)
	$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) *.cu


test:	all  *_test.cpp
# 	$(CPP) -c $(CFLAGS) $(LDFLAGS) tensor_test.cpp
# 	$(CPP) tensor_test.o core.so -o tensor_test
# 	./tensor_test
# 	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) conv_test.cpp
# 	$(CPP) conv_test.o core.so -o conv_test
# 	./conv_test
	$(NVCC) -c $(NVCCFLAGS) $(LDFLAGS) $(CUDALIBS) gpusim_test.cpp
	$(NVCC) $(CUDALIBS) tensor.o gpusim_test.o gpusim.o -o gpusim_test
#	./gpusim_test m0.t kern.t h.t

.PHONY:	clean
clean:	
	rm -f *.o *.so iotest *_test
