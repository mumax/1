include ../../../common.mk
include $(GOROOT)/src/Make.$(GOARCH)

TARG=gpu
CGOFILES=\
	gputil.go\



CGO_LDFLAGS+=$(CURDIR)/../../../core/core.so

# files to be removed by make clean
CLEANFILES+=core.cgo1.go core.cgo2.go core.cgo3.8 core.cgo3.c core.cgo4.c

include $(GOROOT)/src/Make.pkg

# seems to install libsim_libsim.so
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
