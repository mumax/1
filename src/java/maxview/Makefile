all:	maxview.jar

maxview.jar: *.java manifest.mf
	javac *.java
	jar cfm maxview.jar manifest.mf *.class
	cp maxview.jar ../../../bin

clean:
	rm -f *.class maxview.jar
