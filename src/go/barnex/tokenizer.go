package barnex

import (
	"fmt"
)

func Tokenize(str string) {

	for t := newTokenizer(str); t.currchr != -1; t.nextChr() {
		fmt.Print(t.currchr)
	}
	return
}

func newTokenizer(str string) *tokenizer {
	t := new(tokenizer)
	t.str = str

	t.pos = 0
	t.prevchr = -1 // no prevchrious character
	t.currchr = -1
	t.nextchr = -1

	if len(str) > 0 {
		t.currchr = int(str[t.pos])
	}
	if len(str) > 1 {
		t.nextchr = int(str[t.pos+1])
	}

	return t
}

func (t *tokenizer) nextChr() {
	t.prevchr = t.currchr
	t.currchr = t.nextchr
	if len(t.str) > t.pos {
		t.pos++
		t.nextchr = int(t.str[t.pos])
	}
}

type tokenizer struct {
	str                       string
	pos                       int
	prevchr, currchr, nextchr int
}
