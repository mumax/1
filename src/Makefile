all: gpukern.so cpukern.so tensor.so gopkgs javapkgs kernel

gpukern.so: gpukern/*.cu
	make -C gpukern

cpukern.so: cpukern/*.cpp
	make -C cpukern

tensor.so: tensor/*.cpp
	make -C tensor

kernel: tensor.so gpukern.so cpukern.so
	make -C kernel_init

gopkgs:
	make -C go

javapkgs:
	make -C java

test:   all
	make test -C gpukern
	make test -C cpukern
	make test -C go

.PHONY: clean
clean:
	rm -rf *.so *.a
	make clean -C tensor
	make clean -C gpukern
	make clean -C cpukern
	make clean -C go
	make clean -C java
	make clean -C kernel_init
	
