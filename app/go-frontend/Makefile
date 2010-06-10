include ../../common.mk

all:
	make -C tensor
	make install -C tensor
	ln -sf tensor/_obj/tensor.a .

# 	make -C core
# 	make install -C core

	make -C refsh
	ln -sf refsh/_obj/refsh.a .

	make -C sim
	ln -sf sim/_obj/sim.a .

	make -C tensor-main
	make -C kernel-main
	make -C config-main
	make -C draw
	make -C main

.PHONY: clean
clean:
	make clean -C tensor
	make clean -C core
	make clean -C sim
	make clean -C tensor-main
	make clean -C kernel-main
	make clean -C config-main
	make clean -C draw
	make clean -C main

test:	all
	make test -C tensor
#	make test -C core


