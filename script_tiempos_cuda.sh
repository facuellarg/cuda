#!/bin/bash
# g++ blur-effect.cpp -o y `pkg-config --cflags --libs opencv` -lpthread

printf "\n --- Times MPI --- \n"
echo "">times_cuda.in
threads=1
theN=8
while [ $theN -le 1024 ]
do  
	./matrixMult $theN  $threads| tee -a times_cuda.in   
    # ./y hdLancia.jpg blur/hd-blur.jpg $theN $threads | tee -a times/hd-times.in    
    theN=$((theN*2))
done

threads=2
theN=8
while [ $theN -le 1024 ]
do  
   ./matrixMult $theN  $threads| tee -a times_cuda.in   
    theN=$((theN*2))
done

threads=3
theN=8
while [ $theN -le 1024 ]
do  
    ./matrixMult $theN  $threads| tee -a times_cuda.in   
    theN=$((theN*2))
done

threads=4
theN=8
while [ $theN -le 1024 ]
do  
  ./matrixMult $theN  $threads| tee -a times_cuda.in   
    theN=$((theN*2))
done
