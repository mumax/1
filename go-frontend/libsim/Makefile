include $(GOROOT)/src/Make.$(GOARCH)

TARG=libsim
CGOFILES=libsim.go

# linker 'flags', perhaps a bit abused here to link to libsim.o
# when we link against libsim.o, we need to include -lfftw3f
# linking against libsim.so automatically includes libfftw3f.so
CGO_LDFLAGS+=$(SIMROOT)/lib/libfft.so
CGO_LDFLAGS+=-lfftw3f -lfftw3f_threads -lpthread 

# files to be removed by make clean
CLEANFILES+=gosim libsim.cgo1.go libsim.cgo2.go libsim.cgo3.8 libsim.cgo3.c libsim.cgo4.c

include $(GOROOT)/src/Make.pkg

# seems to install libsim_libsim.so
%: install %.go
	$(GC) $*.go
	$(LD) -o $@ $*.$O
