#include "libtensor.h"
#include <string>
#include <iostream>

using namespace std;

void improper_arguments(int argc, char** argv){
  cerr << "Usage: " << argv[0] << " [-a | --ascii | -b | --binary]" << endl;
  exit(1);
}

int main(int argc, char** argv){
  
  bool ascii;
  bool recognized = false;
  
  if(argc == 1){
    ascii = true;
  }
  else if(argc == 2){
    string arg = argv[1];
    if(arg == "-a" || arg == "--ascii"){
      recognized = true;
      ascii = true;
    }
    if(arg == "-b" || arg == "--binary"){
      recognized = true;
      ascii = false;
    }
    if(!recognized){
      improper_arguments(argc, argv);
    }
  }
  else{
    improper_arguments(argc, argv);
  }
}