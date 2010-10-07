#! /bin/bash

# floating point arithmetic
# e.g.: calc.sh 1+1
# prints 2

echo "scale=15; $@" | bc
