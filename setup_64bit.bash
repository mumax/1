#! /bin/bash

OUTPUT=bin/mumax
echo '#! /bin/bash' > $OUTPUT
echo 'export SIMROOT='$(pwd) >> $OUTPUT

# GO
echo 'export GOROOT=$SIMROOT/dep/go' >> $OUTPUT
echo 'export GOOS=linux' >> $OUTPUT
echo 'export GOARCH=amd64' >> $OUTPUT
echo 'export GOBIN=$SIMROOT/bin/go' >> $OUTPUT
echo 'export GOMAXPROCS=$(grep processor /proc/cpuinfo | wc -l) ' >> $OUTPUT
echo 'export PATH=$SIMROOT/bin/go:$PATH ' >> $OUTPUT
echo 'export LD_LIBRARY_PATH=$GOROOT/pkg/$GOOS"_"$GOARCH ' >> $OUTPUT

# cuda
echo 'export LD_LIBRARY_PATH=$SIMROOT/dep/cuda/lib64:$SIMROOT/dep/cuda/lib:$LD_LIBRARY_PATH ' >> $OUTPUT
echo 'export L="-L$SIMROOT/dep/cuda/lib64 -L$SIMROOT/dep/cuda/lib"' >> $OUTPUT
echo 'export C_INCLUDE_PATH=$SIMROOT/dep/cuda/include:$C_INCLUDE_PATH ' >> $OUTPUT
echo 'export CPLUS_INCLUDE_PATH=$SIMROOT/dep/cuda/include:$CPLUS_INCLUDE_PATH ' >> $OUTPUT

# fftw
echo 'export LD_LIBRARY_PATH=$SIMROOT/lib/fftw/lib:$LD_LIBRARY_PATH' >> $OUTPUT
echo 'export L="-L $SIMROOT/lib/fftw/lib"' >> OUTPUT
echo 'export C_INCLUDE_PATH=$SIMROOT/lib/fftw/include:$C_INCLUDE_PATH' >> $OUTPUT
echo 'export CPLUS_INCLUDE_PATH=$SIMROOT/lib/fftw/include:$CPLUS_INCLUDE_PATH' >> $OUTPUT

# simulation
echo 'export PATH=$PATH:$SIMROOT/bin' >> $OUTPUT
echo 'export LD_LIBRARY_PATH=$SIMROOT/lib:$LD_LIBRARY_PATH' >> $OUTPUT
echo 'export L="$L -L$SIMROOT/lib"' >> OUTPUT
echo 'export C_INCLUDE_PATH=$SIMROOT/src/gpukern:$SIMROOT/src/cpukern:$C_INCLUDE_PATH' >> $OUTPUT
echo 'export CPLUS_INCLUDE_PATH=$SIMROOT/src/gpukern:$SIMROOT/src/cpukern:$CPLUS_INCLUDE_PATH' >> $OUTPUT

echo '$SIMROOT/bin/mumax-sim $@' >> $OUTPUT

chmod u+x $OUTPUT

echo You can now run $SIMROOT/bin/mumax to start burning GPU cycles.
echo Consider adding this line to your .bashrc file:
echo 'export PATH=$PATH:'$SIMROOT/bin 
echo You need to re-run this setup script only when you moved this directory.