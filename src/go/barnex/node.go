package barnex

type Node struct {
	Next, Prev *Node
	Parent     *Node
	Child      []*Node
	Text       string
}
