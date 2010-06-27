include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=gpu

CGOFILES=\
	gpu.go\

GOFILES=\
	fft.go\
	conv.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/$(TARG).so

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
