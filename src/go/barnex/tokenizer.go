package barnex



func Tokenize(str string) (root *Node) {
//   t := newTokenizer(str)
  return
}

func newTokenizer(str string) *tokenizer{
  t := new(tokenizer)
  t.str = str

  t.pos = 0
  t.prev = -1 // no previous character
  t.curr = -1
  t.next = -1
  
  if len(str) > 0{
      t.curr = int(str[t.pos])
  }
  if len(str) > 1{
    t.next = int(str[t.pos+1])
  }
  
  return t
}

type tokenizer struct {
  str string
  pos int
  prev, curr, next int
}