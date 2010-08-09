package refsh

import(
  "io"
  "container/vector"
)


type Tokenizer struct{
  io.Reader
  words vector.StringVector
}

func NewTokenizer(in io.Reader) *Tokenizer{
  return &Tokenizer{in, vector.StringVector(make([]string, 10))}
}

func (t *Tokenizer) ReadChar() int{
  switch nr, err := f.Read(buffer); true {
            case nr < 0:   // error
                fmt.FPrintln(os.Stderr, "read error:", err);
                os.Exit(1)
            case nr == 0:  // eof
                return -1
            case nr > 0:   // ok
                return int(buffer[0])
  }
}

func (t *Tokenizer) ReadLine() []string{
  buffer := make([]byte, 1)
        for {
            
            }
        }
}