package main

import(
  "fmt";
  "os";
  . "../sim";
  . "../tensor";
  . "core";
)

var units Units = *NewUnits();
var dt float = 1E-6;
var hExt []float = []float{0., 0., 0.};
var m StoredTensor;
var steps int;
var savem float;
var t float;
var alpha float = 1.0;

func main(){
  commands, args := parseArgs();
  for i:=range(commands){
    exec(commands[i], args[i]);
  }
  run();
}

func run(){
  toInternalUnits();
  size := units.Size;

  demag := FaceKernel(size, units.CellSize);
  exchange := Exch6NgbrKernel(size, units.CellSize);
  kernel := Buffer(Add(exchange, demag));
  solver := NewGpuHeun(size[0], size[1], size[2], kernel, hExt);

  solver.LoadM(m);
  t = savem; // to trigger output for t=0
  for i:=0; i<steps; i++{
    if t >= savem{
      solver.StoreM(m);
      saveM(i);
      t = 0;
    }
    solver.Step(dt, alpha);
    t += dt;
    
  }
  
}

func toInternalUnits(){
  units.AssertValid();
  for i:=range(units.CellSize){
    units.CellSize[i] /= units.UnitLength();
  }
  savem /= units.UnitTime();
  
  for i:=0; i<3; i++{
    hExt[i] /= units.UnitField();
  }

  dt /= units.UnitTime();

  units.PrintInfo(os.Stderr);
}

func saveM(i int){
  fname := fmt.Sprintf("m%07d.t", i);
  WriteFile(fname, m);
}

func exec(command string, args []string){
  switch command{
    case "--size":
	units.Size = parseSize(args);
    case "--cellsize":
	units.CellSize = parseCellSize(args);
    case "--aexch":
	 argCount(command, args, 1, 1);
	 units.AExch = Atof(args[0]);
    case "--msat":
         argCount(command, args, 1, 1);
	 units.MSat = Atof(args[0]);
    case "--alpha":
         argCount(command, args, 1, 1);
	 alpha = Atof(args[0]);
    case "--dt":
	 argCount(command, args, 1, 1);
	 dt = Atof(args[0]);
    case "--hext":
	 argCount(command, args, 3, 3);
	 hExt = parseCellSize(args);
    case "--loadm":
	 argCount(command, args, 1, 1);
	 m = ReadFile(args[0]);
    case "--steps":
	 argCount(command, args, 1, 1);
	 steps = Atoi(args[0]);
    case "--autosavem":
	 argCount(command, args, 1, 1);
	 savem = Atof(args[0]);
    default:
      fmt.Fprintln(os.Stderr, "unknown command:", command);
      os.Exit(-1);
  }
}


func parseSize(args []string) []int{
  argCount("size", args, 3, 3);
  size := make([]int, 3);
  for i:=0; i<3; i++{
    size[i] = Atoi(args[i]);
  }
  return size;
}

func parseCellSize(args []string) []float{
  argCount("size", args, 3, 3);
  size := make([]float, 3);
  for i:=0; i<3; i++{
    size[i] = Atof(args[i]);
  }
  return size;
}
