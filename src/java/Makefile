all:	maxview.jar

maxview.jar:  maxview/*.java maxview/manifest.mf
	javac maxview/*.java
	jar cfm maxview.jar maxview/manifest.mf maxview/*.class
	cp maxview.jar ../../../bin

clean:
	rm -f maxview/*.class maxview.jar
