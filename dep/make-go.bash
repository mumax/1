#! /bin/bash

echo compiling GO
cd go/src
./all.bash
cd ../..

cp $GOROOT/pkg/$GOOS'_'$GOARCH/libcgo.so $SIMROOT/lib
