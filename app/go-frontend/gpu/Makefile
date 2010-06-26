include ../../../common.mk
include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=gpu
CGOFILES=\
	gpu.go\

GOFILES=\
	conv.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

# files to be removed by make clean
CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/$(TARG).so

include $(GOROOT)/src/Make.pkg

# seems to install libsim_libsim.so
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
