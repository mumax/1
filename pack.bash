#! /bin/bash

# Produces a tarball with mumax binaries

echo packing binaries into mumax.tar.gz ...

OUT=mumax

rm -rf $OUT
mkdir $OUT


cp setup_64bit.bash $OUT
cp -r bin $OUT
cp -r lib $OUT


mkdir $OUT/examples
cp -r examples/*.in $OUT/examples


tar cv $OUT | gzip > mumax.tar.gz

#rm -rf $OUT