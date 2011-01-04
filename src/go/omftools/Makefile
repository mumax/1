include $(GOROOT)/src/Make.inc

main:	*.go
	$(GC)  main.go draw.go resize.go
	$(LD)  -o $@ *.$O
	cp -f $(CURDIR)/main ../../../bin/omftool

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
