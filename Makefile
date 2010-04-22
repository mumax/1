# This is the master makefile for the entire project

include common.mk

export SIMROOT=$(CURDIR)

all:
	make -C core
	make -C app

clean:
	make clean -C core
	make clean -C app
	make clean -C sims
	make clean -C bin
	rm -rf doc/*

test:	all
	make test -C core
	make test -C app

