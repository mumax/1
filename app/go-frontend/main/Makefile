include $(GOROOT)/src/Make.$(GOARCH)

main:	*.go
	ln -sf ../kernel-main/flags.go .
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	ln -sf $(CURDIR)/main ../../../bin/gosim

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
