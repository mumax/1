all:
	chmod u+x make-fftw.bash
	./make-fftw.bash
	mkdir cuda
	mkdir cuda/lib
	mkdir cuda/lib64
	cp -r /usr/local/cuda/lib/libcudart.so.3.2.* cuda/lib/libcudart.so.3
	cp -r /usr/local/cuda/lib/libcufft.so.3.2.* cuda/lib/libcufft.so.3
	cp -r /usr/local/cuda/lib/libcurand.so.3.2.* cuda/lib/libcucurand.so.3
	cp -r /usr/local/cuda/lib64/libcudart.so.3.2.* cuda/lib64/libcudart.so.3
	cp -r /usr/local/cuda/lib64/libcufft.so.3.2.* cuda/lib64/libcufft.so.3
	cp -r /usr/local/cuda/lib64/libcurand.so.3.2.* cuda/lib64/libcucurand.so.3

.PHONY: clean
clean:
	make clean -C fftw
