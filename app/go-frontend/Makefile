all:
	make -C tensor
	make install -C tensor
	ln -sf tensor/_obj/tensor.a .
#	make -C core
#	ln -sf core/_obj/core.a .
	make -C sim
	ln -sf sim/_obj/sim.a .
	make -C tensor-main
	make -C kernel-main

.PHONY: clean
clean:
	make clean -C tensor
	make clean -C core
	make clean -C sim
	make clean -C tensor-main
	make clean -C kernel-main

test:	all
	make test -C tensor
#	make test -C core

