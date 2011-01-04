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
	multigpu.go\
	tensor.go\
	fft.go\
	conv.go\
	torque.go\
	solver.go\
	euler.go\
	heun.go\
	rk.go\
	semianal.go\
	kernel.go\
	demag.go\
	exchange.go\
	material.go\
	util.go\
	sim.go\
	simconfig.go\
	simfield.go\
	simspintorque.go\
	simgeom.go\
	edge.go\
	geom.go\
	simmaterial.go\
	simanis.go\
	simoutput.go\
	simrun.go\
	simsolver.go\
	remotedevice.go\
	simserver.go\
	timer.go\
	dashboard.go\
	appliedfield.go\
	mesh.go\
	redux.go\
	adaptive_euler.go\
	adaptive_heun.go\
	main.go\
        daemon.go\
	stdout.go\
	simmetadata.go\
	resample.go\
	error.go\
	wisdom.go\

CGO_CFLAGS+=-I../../../lib/fftw/include/ -I/usr/local/cuda/include/


# When installing, these libraries must be present (at least symlinked) in the current directory
CGO_LDFLAGS+=-L../../../lib -L/usr/local/cuda/lib64  -L/usr/local/cuda/lib libgpukern.so libcpukern.so

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/cgo_$(TARG).so 
	
include $(GOROOT)/src/Make.pkg
