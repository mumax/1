# Common code for all makefiles
# Should be included in each makefile

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
# --compiler-options -DNDEBUG\
# uncomment the above line to disable all assert() statements
