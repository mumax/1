#export SIMROOT=$(pwd)
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SIMROOT/lib
make clean
if make; then
  make test;
fi;
