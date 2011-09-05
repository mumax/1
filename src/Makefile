all: gpukern.so  tensor.so gopkgs javapkgs

gpukern.so: gpukern/*.cu
	make -C gpukern

tensor.so: tensor/*.cpp
	make -C tensor

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
	make clean -C go
	make clean -C java
	
