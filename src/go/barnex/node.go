package barnex

import (
	"go/token"
)


const (
	EOF    = token.EOF
	LPAREN = token.LPAREN
	RPAREN = token.RPAREN
)

type Node struct {
	Next, Prev *Node
	Parent     *Node
	Child      []*Node

	Text string
	Type token.Token
	Pos  token.Position
}


func NewRootNode() *Node {
	t := new(Node)
	t.Type = token.EOF
	return t
}

func NewEOFNode() *Node {
	t := new(Node)
	t.Type = token.EOF
	return t
}


func NewNode(pos token.Position, Type token.Token, lit string) *Node {
	node := new(Node)
	node.Text = lit
	node.Type = Type
	node.Pos = pos
	return node
}


func (a *Node) Append(b *Node) {
	a.Next = b
	b.Prev = a
}

func (n *Node) String() string {
	return n.Text + "\t\t(" + n.Type.String() + ")"
}
