all:
	chmod u+x make-go.bash
	chmod u+x make-fftw.bash
	make -C cuda
	./make-go.bash
	./make-fftw.bash

.PHONY: clean
clean:
	chmod u+x clean-go.bash
	make clean -C fftw
	./clean-go.bash