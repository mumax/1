# todo: we need to express the depencies between the packages

include ../../common.mk

all:
	make install -C tensor
	
# 	make -C core
# 	make install -C core

	make install -C sim/gpu
	make install -C refsh
	make install -C sim
	
	make -C tensor-main
	make -C kernel-main
	make -C config-main
	make -C draw
# 	make -C main

.PHONY: clean
clean:
	make clean -C tensor
	make clean -C core
	make clean -C sim/gpu
	make clean -C sim
	make clean -C tensor-main
	make clean -C kernel-main
	make clean -C config-main
	make clean -C draw
	make clean -C main
	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/core.a
	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/core.so


test:	all
	make test -C tensor
	make test -C core


