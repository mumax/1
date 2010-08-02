all: gpukern.so cpukern.so gopkgs

gpukern.so: gpukern/*.cu
	make -C gpukern

cpukern.so: cpukern/*.cpp
	make -C cpukern

gopkgs:
	make -C go

.PHONY: clean
clean:
	rm -rf *.so *.a
	make clean -C gpukern
	make clean -C cpukern
	make clean -C go