#! /bin/bash

cd go
./src/clean.bash
rm -f */*.o
rm -f */*/*.o
rm -f */*/*/*.o
rm -f */*/*/*/*.o
rm -f */*/*/*/*/*.o
cd ..
