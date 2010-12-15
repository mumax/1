include ../../../common.mk
include $(GOROOT)/src/Make.inc

CLEANFILES+=*.t *.png *.jpg

main:	*.go
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	cp -f $(CURDIR)/main $(CURDIR)/../../../bin/mumax-sim
	cp -f $(GOROOT)/pkg/$(GOOS)_$(GOARCH)/cgo_sim.so ../../../lib

.PHONY:	clean
clean:
	rm -f  main *.a *.$O $(CLEANFILES)
