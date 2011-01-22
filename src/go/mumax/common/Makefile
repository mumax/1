include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/common

GOFILES=\
	error.go\
	constants.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
