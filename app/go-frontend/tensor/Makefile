include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=tensor

CGOFILES=native.go

GOFILES=\
	assert.go\
	arrayslice.go\
	iterator.go\
	io.go\
	tensor.go\
	tensorops.go\
	vector.go\
	storedtensor.go\
	out.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a $(PKGDIR)/$(TARG).so iotest

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
