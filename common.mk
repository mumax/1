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