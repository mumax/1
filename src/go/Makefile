all:
	gomake -C mumax
	gomake install -C refsh
	gomake -C omftools
	gomake -C odttools
	gomake -C template
	gomake -C maxtest
	cp -f ../../lib/libcpukern.so sim/
	cp -f ../../lib/libgpukern.so sim/
	gomake -C sim
	gomake install -C sim
	gomake -C main
	gomake -C kernel-main

.PHONY: clean
clean:
	gomake clean -C mumax
	gomake clean -C omftools
	gomake clean -C odttools
	gomake clean -C template
	gomake clean -C sim
	gomake clean -C refsh
	gomake clean -C maxtest
	gomake clean -C outtools
	gomake clean -C tensor-main
	gomake clean -C config-main
	gomake clean -C main
	gomake clean -C kernel-main

	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/sim.a
	rm -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/cgo_sim.so
	

test:	all
	gomake test -C tensor2
	gomake test -C tensor
	gomake test -C sim


