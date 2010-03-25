# The compilers:
CC=gcc
CPP=g++
NVCC=nvcc

CFLAGS+=\
  -Werror\
  -fPIC\
  -g\


NVCCFLAGS+=\
  --compiler-options -fPIC\
  --compiler-options -g\
