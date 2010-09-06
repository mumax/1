include $(GOROOT)/src/Make.inc

main:	*.go
	ln -sf ../tensor-main/flags.go .
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O
	ln -sf $(CURDIR)/main ../../../bin/config

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
