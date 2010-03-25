# The compilers:
export CC=gcc
export CPP=g++
export NVCC=nvcc

export CFLAGS+=\
  -Werror\
  -fPIC\
  -g\

export NVCCFLAGS+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\