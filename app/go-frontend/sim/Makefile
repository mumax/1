include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=sim

GOFILES=\
	kernel.go\
	demag.go\
	exchange.go\
	units.go\
	config.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a
	
include $(GOROOT)/src/Make.pkg
