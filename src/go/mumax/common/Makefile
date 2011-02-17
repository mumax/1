include $(GOROOT)/src/Make.inc
PKGDIR=$(GOROOT)/pkg/$(GOOS)_$(GOARCH)

TARG=mumax/common

GOFILES=\
	assert.go\
	error.go\
	constants.go\
	io.go\
	blockingreader.go\
	misc.go\
	cufftsize.go\
	print.go\
	readline.go\
	strconv.go\

CLEANFILES+=$(PKGDIR)/$(TARG).a

include $(GOROOT)/src/Make.pkg

%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
