include ../common.mk


all:
	make -C cpp-frontend
	make -C go-frontend

clean:
	make clean -C cpp-frontend
	make clean -C go-frontend

test:	all
	make test -C cpp-frontend
	make test -C go-frontend

