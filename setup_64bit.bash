#! /bin/bash

OUTPUT=tmpfile
echo '#! /bin/bash' > $OUTPUT
echo 'export SIMROOT='$(pwd) >> $OUTPUT

# GO
echo 'export GOMAXPROCS=$(grep processor /proc/cpuinfo | wc -l) ' >> $OUTPUT

# cuda
echo 'export LD_LIBRARY_PATH=$SIMROOT/dep/cuda/lib64:$SIMROOT/dep/cuda/lib:$LD_LIBRARY_PATH ' >> $OUTPUT

# fftw
echo 'export LD_LIBRARY_PATH=$SIMROOT/lib/fftw/lib:$LD_LIBRARY_PATH' >> $OUTPUT

# python
echo 'export PYTHONPATH=$PYTHONPATH:$SIMROOT/lib' >> $OUTPUT

# simulation
echo 'export PATH=$PATH:$SIMROOT/bin' >> $OUTPUT
echo 'export LD_LIBRARY_PATH=$SIMROOT/lib:$LD_LIBRARY_PATH' >> $OUTPUT

# wrappers around executables
cp $OUTPUT bin/mumax
echo '$SIMROOT/bin/mumax-sim $@' >> bin/mumax
chmod u+x bin/mumax

cp $OUTPUT bin/maxview
echo 'java -jar $SIMROOT/bin/maxview.jar' >> bin/maxview
chmod u+x bin/maxview

cp $OUTPUT bin/maxview-x
echo 'omftool $@ --draw3d-dump | java -jar $SIMROOT/bin/maxview.jar' >> bin/maxview-x
chmod u+x bin/maxview-x

rm -f $OUTPUT

echo You can now run $SIMROOT/bin/mumax to start burning GPU cycles.
echo Consider adding this line to your .bashrc file:
echo 'export PATH=$PATH:'$SIMROOT/bin 
echo You need to re-run this setup script only when you moved this directory.
