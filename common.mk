# Common code for all makefiles
# Should be included in each makefile

# The C compiler
CC=gcc

# The C++ compiler
CPP=g++

# The CUDA compiler
NVCC=nvcc

# Flags to be passed to CC and CPP
CFLAGS+=\
  -Wall\
  -Werror\
  -fPIC\
  -g\

# Flags to be passed to NVCC
NVCCFLAGS+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
# --compiler-options -DNDEBUG\
# -DNDEBUG disables all assert() statements

# Cuda libraries
#CUDALIBS= -l:libcudart.so -l:libcufft.so
CUDALIBS= -lcudart -lcufft


