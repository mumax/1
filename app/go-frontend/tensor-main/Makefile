include $(GOROOT)/src/Make.$(GOARCH)

main:	*.go
	$(GC) -I_obj main.go
	$(LD) -L_obj -o $@ main.$O
	ln -sf $(SIMROOT)/go-frontend/tensor-main/main $(SIMROOT)/bin/tensor

.PHONY:	clean
clean:
	rm -f  main *.a *.$O
