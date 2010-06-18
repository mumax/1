#! /bin/bash

make clean;
sed 's/common.mk/emu.mk/g' Makefile > Makefile-emu;
make -f Makefile-emu $@
rm -f Makefile-emu
