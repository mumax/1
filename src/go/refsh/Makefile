include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=refsh

GOFILES=\
	refsh.go\
	caller.go\
	resolve.go\
	parse.go\
	readline.go\
	flags.go\
	flags2.go\
	
CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg
