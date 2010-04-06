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
CLEANFILES+=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)/tensor_native.so iotest

include $(GOROOT)/src/Make.pkg

# seems to install .so library
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
