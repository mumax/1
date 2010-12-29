all:	maxview.jar

maxview.jar:  refsh.jar maxview/*.java maxview/manifest.mf
	javac maxview/*.java
	jar cfm maxview.jar maxview/manifest.mf refsh/*.class maxview/*.class
	cp maxview.jar ../../bin

refsh.jar:	refsh/*.java
	javac refsh/*.java
	jar cf refsh.jar refsh/*.class
clean:
	rm -f maxview/*.class maxview.jar refsh/*.class refsh.jar
