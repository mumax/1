# This is the master makefile for the entire project

include common.mk

export SIMROOT=$(CURDIR)

all:
	make -C libtensor
	make -C libfft
	make -C go-frontend

clean:
	make clean -C libtensor
	make clean -C libfft
	make clean -C go-frontend

test:	all
	make test -C libtensor
	make test -C libfft
	make test -C go-frontend
