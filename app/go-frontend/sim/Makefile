include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=sim

CGOFILES=\
	gpu.go

GOFILES=\
	device.go\
	tensor.go\
	backend.go\
	fft.go\
	conv.go\
	field.go\
	solver.go\
	euler.go\
	kernel.go\
	demag.go\
	exchange.go\
	units.go\
	config.go\
	assert.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

CLEANFILES+=$(PKGDIR)/$(TARG).a
	
include $(GOROOT)/src/Make.pkg
