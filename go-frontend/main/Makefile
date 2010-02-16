include $(GOROOT)/src/Make.$(GOARCH)

main:
	$(GC) -I_obj main.go
	$(LD) -L_obj -o $@ main.$O

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
