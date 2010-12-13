#! /bin/bash

# Produces a tarball with mumax binaries

echo packing binaries into mumax.tar.gz ...

OUT=mumax

rm -rf $OUT
mkdir $OUT


cp setup_64bit.bash $OUT
cp -r bin $OUT
cp -r lib $OUT
#cp -r dep/cuda $OUT/dep
#cp -r dep/go $OUT/dep

rm -rf $OUT/dep/go/src
rm -rf $OUT/dep/go/test
rm -rf $OUT/bin/go
rm -rf $OUT/dep/fftw
rm -rf $OUT/dep/cuda/devdriver*
rm -rf $OUT/dep/cuda/include

mkdir $OUT/examples
cp -r examples/*.in $OUT/examples


tar cv $OUT | gzip > mumax.tar.gz

#rm -rf $OUT