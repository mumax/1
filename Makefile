# This is the master makefile for the entire project

include common.mk

export SIMROOT=$(CURDIR)

all:
	make -C core

clean:
	make clean -C core

test:	all
	make test -C core

