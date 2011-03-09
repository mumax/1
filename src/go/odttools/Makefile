include $(GOROOT)/src/Make.inc

main:	*.go
	$(GC)  main.go desc.go peak.go filter.go speed.go
	$(LD)  -o $@ *.$O
	cp -f $(CURDIR)/main ../../../bin/odttool

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
