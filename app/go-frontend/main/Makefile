include ../../../common.mk
include $(GOROOT)/src/Make.$(GOARCH)

CLEANFILES+=*.t *.png

main:	*.go
	ln -sf ../kernel-main/flags.go .
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O

.PHONY:	clean
clean:
	rm -f  main *.a *.$O $(CLEANFILES)
