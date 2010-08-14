package refsh

import(
  "reflect"
)

// Caller unifies anything that can be called:
// a Method or a reflect.FuncValue
type Caller interface{
  // Call the thing
  Call(args []reflect.Value) []reflect.Value
  // Types of the input parameters
  In(i int) reflect.Type
  // Number of input parameters
  NumIn() int
}



type AMethod struct{
  reciever reflect.Value
  function *reflect.FuncValue
}

func(m *AMethod) Call(args []reflect.Value) []reflect.Value{
  methargs := make([]reflect.Value, len(args)+1)  // todo: buffer in method struct
  methargs[0] = m.reciever
  for i, arg := range(args){
    methargs[i+1] = arg
  }
  return m.function.Call(methargs)
}

func(m *AMethod) In(i int) reflect.Type{
  return (m.function.Type().(*reflect.FuncType)).In(i+1)    // do not treat the reciever (1st argument) as an actual argument
}

func(m *AMethod) NumIn() int{
  return (m.function.Type().(*reflect.FuncType)).NumIn() - 1 // do not treat the reciever (1st argument) as an actual argument
}


type AFunction reflect.FuncValue

func(f *AFunction) In(i int) reflect.Type{
  return (*reflect.FuncValue)(f).Type().(*reflect.FuncType).In(i)
}

func(f *AFunction) NumIn() int{
  return (*reflect.FuncValue)(f).Type().(*reflect.FuncType).NumIn()
}

func(f *AFunction) Call(args []reflect.Value) []reflect.Value{
  return (*reflect.FuncValue)(f).Call(args)
}

