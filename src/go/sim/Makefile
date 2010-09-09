include ../../../common.mk
include $(GOROOT)/src/Make.inc
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
	euler.go\
	heun.go\
	semianal.go\
	kernel.go\
	demag.go\
	exchange.go\
	material.go\
	magnet.go\
	util.go\
	sim.go\
	simbackend.go\
	simconfig.go\
	simfield.go\
	simgeom.go\
	simmaterial.go\
	simoutput.go\
	simrun.go\
	simsolver.go\
	draw.go\
	remotedevice.go\
	simserver.go\
	timer.go\
	dashboard.go\
	appliedfield.go\
	mesh.go\
	input.go\

CGO_LDFLAGS+=$(CURDIR)/../../../lib/libgpukern.so $(CURDIR)/../../../lib/libcpukern.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
