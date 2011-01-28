include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=tensor

GOFILES=\
	tensor.go\
	arrayslice.go\
	iterator.go\
	util.go\
	vector.go\
	input.go\
	output.go\
	math.go\
	legacyio.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
