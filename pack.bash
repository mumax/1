#! /bin/bash

# Produces a tarball with mumax binaries

echo packing binaries into mumax.tar.gz ...

OUT=mumax

rm -rf $OUT
mkdir $OUT


cp doc/manual/manual.pdf $OUT
cp doc/api/latex/refman.pdf $OUT/python_api.pdf
ln -s doc/api/html/namespacemumax.html $OUT/python_api.html
cp setup.bash $OUT
cp LICENSE.txt $OUT
cp README.txt $OUT
cp Makefile $OUT
cp common.mk $OUT
cp -r bin $OUT
cp -r lib $OUT
cp -r doc $OUT

mkdir $OUT/src
cp -r src/ $OUT
rm -rf $OUT/src/dev_arne
rm -rf $OUT/src/dev_ben

mkdir $OUT/examples
cp -r examples/*.in $OUT/examples
cp -r examples/*.py $OUT/examples


tar cv $OUT | gzip > mumax.tar.gz

#rm -rf $OUT
