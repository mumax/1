all:
	make -C libtensor
	make -C libfft
	make -C go-frontend
	make -C bin

clean:
	make clean -C libtensor
	make clean -C libfft
	make clean -C go-frontend
	make clean -C bin

test:
	make test -C go-frontend
