include $(GOROOT)/src/Make.inc

main:	*.go
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	cp -f $(CURDIR)/main ../../../bin/omftool

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
