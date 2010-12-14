# todo: we need to express the depencies between the packages

include ../../common.mk

all:
	gomake install -C iotool
	gomake install -C tensor
	gomake install -C refsh
	ln -s $(SIMROOT)/lib/libcpukern.so sim/
	ln -s $(SIMROOT)/lib/libgpukern.so sim/
	ln -s $(SIMROOT)/lib/libcpukern.a sim/
	ln -s $(SIMROOT)/lib/libgpukern.a sim/

	gomake install -C sim

	gomake -C outtools
# 	make -C tensor-main
# 	make -C kernel-main
# 	make -C config-main
# 	make -C draw
	gomake -C main

.PHONY: clean
clean:
	gomake clean -C iotool
	gomake clean -C tensor
	gomake clean -C sim
	gomake clean -C refsh

	gomake clean -C outtools
	gomake clean -C tensor-main
	gomake clean -C config-main
	gomake clean -C draw
	gomake clean -C main

	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/sim.a
	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/cgo_sim.so
	


test:	all
	gomake test -C tensor2
	gomake test -C tensor
	gomake test -C sim


