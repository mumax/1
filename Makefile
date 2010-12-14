
all:
	make -C src
	make -C bin

depend:
	make -C dep

.PHONY: clean
clean:
	make clean -C bin
	make clean -C src
	make clean -C dep
	make clean -C lib
	
doc:
	make -C doc