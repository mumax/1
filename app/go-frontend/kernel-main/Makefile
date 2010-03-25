include $(GOROOT)/src/Make.$(GOARCH)

main:	*.go
	$(GC) -I_obj *.go
	$(LD) -L_obj -o $@ *.$O

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
