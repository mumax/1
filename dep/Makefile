all:
	chmod u+x make-fftw.bash
	make -C cuda
	./make-fftw.bash

.PHONY: clean
clean:
	make clean -C fftw