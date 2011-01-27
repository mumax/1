PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

all: _common $(PKGDIR)/mumax/device.a $(PKGDIR)/mumax/device/gpu.a $(PKGDIR)/mumax/device/cpu.a

$(PKGDIR)/mumax/device/gpu.a: _common device/gpu/*.go
	cp -f ../../../lib/libgpukern.so device/gpu
	gomake -C device/gpu
	gomake install -C device/gpu

$(PKGDIR)/mumax/device/cpu.a: _common device/cpu/*.go
	cp -f ../../../lib/libcpukern.so device/cpu
	gomake -C device/cpu
	gomake install -C device/cpu

$(PKGDIR)/mumax/device.a:	_common device/*.go
	gomake -C device
	gomake install -C device
	
_common: $(PKGDIR)/mumax/common.a

$(PKGDIR)/mumax/common.a: common/*.go
	gomake -C common
	gomake install -C common

	
clean:
	gomake clean -C common
	gomake clean -C device
	gomake clean -C device/cpu
	gomake clean -C device/gpu

