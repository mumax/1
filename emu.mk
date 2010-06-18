# Can be included in a makefile (instead of common.mk) to enable device emulation

CC=gcc
CPP=g++
NVCC=nvcc

CFLAGS+=\
  -Wall\
  -Werror\
  -fPIC\
  -g\


NVCCFLAGS+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --compiler-options -g\
  -G\
  --device-emulation\
# --compiler-options -DNDEBUG\


# -DNDEBUG disables all assert() statements
# --device-emulation allows to run it on a CPU, requires "emu" to be appended to all cuda library names (e.g. -lcudartemu)


#remove the "emu" when not using device emulation
CUDALIBS= -lcudartemu -lcufftemu