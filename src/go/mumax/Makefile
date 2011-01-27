all: common.a device.a gpu.a cpu.a

gpu.a: common device/gpu/*.go
	cp -f ../../../lib/libgpukern.so device/gpu
	gomake -C device/gpu
	gomake install -C device/gpu

cpu.a: common device/cpu/*.go
	cp -f ../../../lib/libcpukern.so device/cpu
	gomake -C device/cpu
	gomake install -C device/cpu

device.a:	common device/*.go
	gomake -C device
	gomake install -C device
	
common.a: common/*.go
	gomake -C common
	gomake install -C common

	
clean:
	gomake clean -C common
	gomake clean -C device
	gomake clean -C device/cpu
	gomake clean -C device/gpu

