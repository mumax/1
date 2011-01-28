include ../../../../../common.mk
include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/device/gpu

CGOFILES=\
	gpu.go\

GOFILES=\

CGO_CFLAGS+=-I/usr/local/cuda/include/


# When installing, these libraries must be present (at least symlinked) in the current directory
CGO_LDFLAGS+=-L../../../lib -L/usr/local/cuda/lib  -L/usr/local/cuda/lib64 libgpukern.so 

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
