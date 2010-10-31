package barnex

import (
	"fmt"
)

func Tokenize(str string) {

	for t := newTokenizer(str); t.currchr != -1; t.nextChr() {
		fmt.Print(string(byte(t.currchr)))
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

// Advances the tokenizer to the next character,
// which can be retrieved with tokenizer.currchr
func (t *tokenizer) nextChr() {
	t.prevchr = t.currchr
	t.currchr = t.nextchr
	if len(t.str) > t.pos+2 {
		t.pos++
		t.nextchr = int(t.str[t.pos+1])
	} else {
		t.nextchr = -1
	}
}

type tokenizer struct {
  
	str              string // string being tokenized
	
	pos              int    // position of currchr in string
	currchr          int    // current character
	prevchr, nextchr int    // previous/next character

	
}
