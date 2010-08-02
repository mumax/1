#! /bin/bash

#This script recompiles the entire suite

if ! source ./bin/setenv; then
  echo you need to run ./setup_XXbit.bash first
  exit
fi

# Now we can start with the compilation

make