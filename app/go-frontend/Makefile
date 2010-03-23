all:
	make install -C core
# 	make -C assert
# 	ln -sf assert/_obj/assert.a .
# 	make -C util
# 	ln -sf util/_obj/util.a .
# 	make install -C libsim
# 	ln -sf libsim/_obj/libsim.a .
# 	make -C tensor
# 	ln -sf tensor/_obj/tensor.a .
# 	make -C fft
# 	ln -sf fft/_obj/fft.a .
# 	make -C sim
# 	ln -sf sim/_obj/sim.a .
# 	make -C main
# 	ln -sf main/main ./run
# 	make -C tensor-main

.PHONY: clean
clean:
	make clean -C core
# 	make clean -C assert
# 	make clean -C util
# 	make clean -C libsim
# 	make clean -C tensor
# 	make clean -C fft
# 	make clean -C sim
# 	make clean -C main
# 	make clean -C tensor-main
# 	rm -f *.a run

test:
	make test -C core
# 	make test -C tensor
# 	make test -C fft
# 	make test -C sim
