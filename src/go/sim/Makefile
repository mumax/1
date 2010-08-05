include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=sim

CGOFILES=\
	gpu.go\
	cpu.go\

GOFILES=\
	device.go\
	backend.go\
	tensor.go\
	fft.go\
	conv.go\
	field.go\
	solver.go\
	timestep.go\
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

CGO_LDFLAGS+=$(CURDIR)/../../../lib/libgpukern.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
