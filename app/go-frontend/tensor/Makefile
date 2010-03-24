include $(GOROOT)/src/Make.$(GOARCH)

TARG=tensor
CGOFILES=native.go

GOFILES=\
	assert.go\
	arrayslice.go\
	iterator.go\
	io.go\
	tensor.go\
	vector.go\
	storedtensor.go\
	out.go\


# files to be removed by make clean
#CLEANFILES+=core.cgo1.go core.cgo2.go core.cgo3.8 core.cgo3.c core.cgo4.c

include $(GOROOT)/src/Make.pkg

# seems to install libsim_libsim.so
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
