include $(GOROOT)/src/Make.$(GOARCH)
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=refsh

GOFILES=\
	refsh.go\
	caller.go\
	resolve.go\
	parse.go\
	readline.go\
	flags.go\
	
CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg
