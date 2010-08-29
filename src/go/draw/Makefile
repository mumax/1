include $(GOROOT)/src/Make.inc

main:	*.go
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	ln -sf $(CURDIR)/main ../../../bin/draw

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
