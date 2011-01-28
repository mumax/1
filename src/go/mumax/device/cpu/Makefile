include ../../../../../common.mk
include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/device/cpu

CGOFILES=\
	cpu.go\

GOFILES=\


CGO_CFLAGS+=-I../../../../../lib/fftw/include/

# When installing, these libraries must be present (at least symlinked) in the current directory
CGO_LDFLAGS+=libcpukern.so


CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
