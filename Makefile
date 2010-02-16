all:
	make -C libfft
	make -C go-frontend

clean:
	make clean -C libfft
	make clean -C go-frontend

test:
	make test -C go-frontend
