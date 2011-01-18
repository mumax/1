include $(GOROOT)/src/Make.inc

main:	*.go
	$(GC)  main.go 
	$(LD)  -o $@ *.$O
	cp -f $(CURDIR)/main ../../../bin/maxtest

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
