include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=gpu

CGOFILES=\
	gpu.go\

GOFILES=\
	tensor.go\
	fft.go\
	conv.go\
	field.go\
	solver.go\
	euler.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/$(TARG).so

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
