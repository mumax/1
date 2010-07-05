include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=sim

CGOFILES=\
	gpu.go

GOFILES=\
	device.go\
	backend.go\
	tensor.go\
	fft.go\
	conv.go\
	field.go\
	solver.go\
	euler.go\
	heun.go\
	semianal.go\
	kernel.go\
	demag.go\
	exchange.go\
	material.go\
	magnet.go\
	config.go\
	util.go\

CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

CLEANFILES+=$(PKGDIR)/$(TARG).a
	
include $(GOROOT)/src/Make.pkg
