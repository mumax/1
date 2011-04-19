PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

all: _common\
	 $(PKGDIR)/mumax/omf.a\
	 $(PKGDIR)/mumax/draw.a\
	 #$(PKGDIR)/mumax/device.a\
	 #$(PKGDIR)/mumax/device/gpu.a\
	 #$(PKGDIR)/mumax/device/cpu.a\
	 #$(PKGDIR)/mumax/device/multigpu.a\

$(PKGDIR)/mumax/device/gpu.a: _common device/gpu/*.go
	cp -f ../../../lib/libgpukern.so device/gpu
	gomake -C device/gpu
	gomake install -C device/gpu

$(PKGDIR)/mumax/device/cpu.a: _common device/cpu/*.go
	cp -f ../../../lib/libcpukern.so device/cpu
	gomake -C device/cpu
	gomake install -C device/cpu

$(PKGDIR)/mumax/device/multigpu.a: _common device/multigpu/*.go
	gomake -C device/multigpu
	gomake install -C device/multigpu

$(PKGDIR)/mumax/device.a: _common device/*.go
	gomake -C device
	gomake install -C device
	
$(PKGDIR)/mumax/omf.a: _common omf/*.go
	gomake -C omf
	gomake install -C omf

$(PKGDIR)/mumax/draw.a: _common draw/*.go
	gomake -C draw
	gomake install -C draw

_common: $(PKGDIR)/mumax/common.a\
	 $(PKGDIR)/mumax/tensor.a\

$(PKGDIR)/mumax/common.a: common/*.go
	gomake -C common
	gomake install -C common

$(PKGDIR)/mumax/tensor.a: tensor/*.go
	gomake -C tensor
	gomake install -C tensor

clean:
	gomake clean -C common
	gomake clean -C tensor
	gomake clean -C device
	gomake clean -C device/cpu
	gomake clean -C device/gpu
	gomake clean -C omf
	gomake clean -C draw

test: cputest gputest tensortest

cputest: 
	echo cpu > /tmp/mumax_test_device
	gomake test -C device

gputest:
	echo gpu > /tmp/mumax_test_device
	gomake test -C device

tensortest:
	gomake test -C tensor

bench: 
	gomake bench -C device
	gomake bench -C tensor

