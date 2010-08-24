include ../../../common.mk
include $(GOROOT)/src/Make.$(GOARCH)

main:	*.go
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	ln -sf $(CURDIR)/main $(CURDIR)/../../../bin/simserver

.PHONY:	clean
clean:
	rm -f  main *.a *.$O $(CLEANFILES)
