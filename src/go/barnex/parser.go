package barnex

func Parse(fname string) (start, stop *Node) {
	start, stop = Tokenize(fname)
  parse(start, stop)
  // TODO root = ...
  return
}

// parses everything BETWEEN start and stop,
// excluding start and stop
func parse(start, stop *Node) {
	parseBrackets(start, stop)
}

func parseBrackets(start, stop *Node) {
	curr := start.Next

	// scan for opening parenthesis
	for curr.Type != EOF && curr.Type != LPAREN{
    curr = curr.Next
  }

  // no opening parenthesis found: no work to do
  if curr.Type == EOF{
    return
  }

  assert(curr.Type == LPAREN)

  // scan for opening parenthesis
  for curr.Type != EOF && curr.Type != LPAREN{
    curr = curr.Next
  }

  // no opening parenthesis found: no work to do
  if curr.Type == EOF{
    return
  }

}


func assert(test bool){
  if !test{
    panic("Assertion failed")
  }
}