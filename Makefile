# Be sure to source bin/setenv before running this makefile.
# You can run ./recompile.bash which will do just that,
# or add the following lines to your ~/.bashrc file (without the '#', of course)

# source /path/to/bin/setenv
# PATH=$PATH:/path/to/bin

# where you replace /path/to/bin with the actual path of the bin/ directory


all:
	make -C src
	make -C bin
	make -C doc

depend:
	make -C lib
	make -C dep

.PHONY: clean
clean:
	make clean -C bin
	make clean -C src
	make clean -C dep
	make clean -C lib
	
doc:
	make -C doc
