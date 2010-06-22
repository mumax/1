include ../../../common.mk
include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=core
CGOFILES=\
	core.go\
	gpuheun.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so -I/usr/local/cuda/include/

# files to be removed by make clean
CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg

# seems to install libsim_libsim.so
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
