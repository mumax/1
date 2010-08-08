all: libcudart.so libcufft.so

libcudart.so: libcudart.so.3
	ln -sf libcudart.so.3 libcudart.so

libcudart.so.3: libcudart.so.3.0.14
	ln -sf libcudart.so.3.0.14 libcudart.so.3

libcufft.so: libcufft.so.3
	ln -sf libcufft.so.3 libcufft.so

libcufft.so.3: libcufft.so.3.0.14
	ln -sf libcufft.so.3.0.14 libcufft.so.3

clean:
	rm -rf libcu*.so libcu*.so.3