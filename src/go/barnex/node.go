package barnex

import (
	"go/token"
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
	return new(Node)
}

func NewNode(pos token.Position, Type token.Token, lit string) *Node {
	node := new(Node)
	node.Text = lit
	node.Type = Type
	node.Pos = pos
	return node
}

func (n *Node) String() string {
	return n.Text + "\t\t(" + n.Type.String() + ")"
}
