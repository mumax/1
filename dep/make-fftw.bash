#! /bin/bash

echo compiling FFTW
cd fftw
make clean
./configure CFLAGS="-fPIC" --enable-float  --enable-threads --enable-sse --prefix $(pwd)/../../lib/fftw
make -j $GOMAXPROCS
make install
cd .. 
