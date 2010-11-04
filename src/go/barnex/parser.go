package barnex

func Parse(fname string) (root *Node) {
	root = Tokenize(fname)

}

func parse(start, stop *Node) {
	parseBrackets(start, stop)
}

func parseBrackets(start, stop *Node) {
	current := start

}
