.PHONY: clean

all:
	cp /usr/local/cuda/lib64/libcudart.so.3 ./libcudart.so.3 || echo;
	cp /usr/local/cuda/lib64/libcufft.so.3 ./libcufft.so.3 || echo;
	cp /usr/local/cuda/lib64/libcurand.so.3 ./libcurand.so.3 || echo;
	
clean:
	rm -rf *.so.* *.a libcudart.* libcufft.* mumax.pyc
