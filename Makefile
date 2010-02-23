# This is the master makefile for the entire project

# The C and C++ compilers:
export CC=gcc
export CPP=g++

export CFLAGS+=\
  -Werror\
  -fPIC\
  -g\

export LDFLAGS+=\

export LD_FFTW_LIBS+=\
  -lfftw3f\
  -lfftw3f_threads\
  -lpthread\

export CLEANFILES+=\
  *.o\
  *.so\
  *.a\

export SIMROOT=$(CURDIR)

all:
	make -C libtensor
	make -C libfft
	make -C go-frontend

clean:
	make clean -C libtensor
	make clean -C libfft
	make clean -C go-frontend

test:
	make test -C libtensor
	make test -C libfft
	make test -C go-frontend
