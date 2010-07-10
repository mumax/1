package refsh

import(
  "io"
  "os"
  "fmt"
)


type Tokenizer struct{
  io.Reader
  buffer []byte
  nextChar byte
}


func NewTokenizer(in io.Reader) *Tokenizer{
  t := new(Tokenizer)
  t.Reader = in
  t.buffer = make([]byte, 1)
  t.nextChar = 0            // will be discarded
  _ = t.readChar()          // initializes nextChar
  return t
}




// func(t *Tokenizer) ReadToken() string{
//   
// }


// func wordChar(chr byte) bool{
//   
// }   


// peeks ahead 
func(t *Tokenizer) peekChar() byte{
  return t.nextChar
}

// reads and returns one character
func(t *Tokenizer) readChar() byte{
  curChar := t.nextChar
  
  switch n, err := t.Read(t.buffer); true {
    
    case n < 0:     // error
      fmt.Fprintln(os.Stderr, "read error:", err)
      os.Exit(1)
      
    case n == 0: 
      t.nextChar = 0     // empty string means EOF
      
    case n > 0:     
      t.nextChar = t.buffer[0]
  }  
  return curChar
}



