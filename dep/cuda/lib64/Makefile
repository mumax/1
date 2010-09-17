all: libcudart.so libcufft.so

libcudart.so: libcudart.so.3
	ln -sf libcudart.so.3 libcudart.so

libcudart.so.3: libcudart.so.3.1.9
	ln -sf libcudart.so.3.1.9 libcudart.so.3

libcufft.so: libcufft.so.3
	ln -sf libcufft.so.3 libcufft.so

libcufft.so.3: libcufft.so.3.1.9
	ln -sf libcufft.so.3.1.9 libcufft.so.3

clean:
	rm -rf libcu*.so libcu*.so.3