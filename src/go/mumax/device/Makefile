include ../../../../common.mk
include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/device

CGOFILES=\
	gpu.go\
	cpu.go\

GOFILES=\
	device.go\
	interface.go\
	multigpu.go\
	tensor.go\

CGO_CFLAGS+=-I../../../../lib/fftw/include/ -I/usr/local/cuda/include/


# When installing, these libraries must be present (at least symlinked) in the current directory
CGO_LDFLAGS+=-L../../../lib -L/usr/local/cuda/lib  -L/usr/local/cuda/lib64 libgpukern.so libcpukern.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
