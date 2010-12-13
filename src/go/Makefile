# todo: we need to express the depencies between the packages

include ../../common.mk

all:
	make install -C iotool
	make install -C tensor
	make install -C refsh
	ln -s $(SIMROOT)/lib/libcpukern.so sim/
	ln -s $(SIMROOT)/lib/libgpukern.so sim/
	ln -s $(SIMROOT)/lib/libcpukern.a sim/
	ln -s $(SIMROOT)/lib/libgpukern.a sim/

	make install -C sim

	make -C outtools
# 	make -C tensor-main
# 	make -C kernel-main
# 	make -C config-main
# 	make -C draw
	make -C main

.PHONY: clean
clean:
	make clean -C iotool
	make clean -C tensor
	make clean -C sim
	make clean -C refsh

	make clean -C outtools
	make clean -C tensor-main
	make clean -C config-main
	make clean -C draw
	make clean -C main

	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/sim.a
	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/cgo_sim.so
	


test:	all
	make test -C tensor2
	make test -C tensor
	make test -C sim


