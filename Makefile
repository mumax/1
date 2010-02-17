all:
	make -C libtensor
	make -C libfft
	make -C go-frontend

clean:
	make clean -C libtensor
	make clean -C libfft
	make clean -C go-frontend

test:
	make test -C go-frontend
