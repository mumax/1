all:
	gomake -C mumax
	gomake install -C iotool
	gomake install -C tensor
	gomake install -C omf
	gomake install -C draw
	gomake install -C refsh
	gomake -C omftools
	gomake -C template
	gomake -C maxtest
	cp -f ../../lib/libcpukern.so sim/
	cp -f ../../lib/libgpukern.so sim/
	gomake -C sim
	gomake install -C sim
#gomake -C outtools
	gomake -C main

.PHONY: clean
clean:
	gomake clean -C iotool
	gomake clean -C omftools
	gomake clean -C template
	gomake clean -C tensor
	gomake clean -C sim
	gomake clean -C refsh
	gomake clean -C maxtest
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


