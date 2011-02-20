include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/tensor

GOFILES=\
	tensor.go\
	interface.go\
	arrayslice.go\
	iterator.go\
	vector.go\
	math.go\
	input.go\
	output.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
