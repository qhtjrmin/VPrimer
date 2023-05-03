#!/bin/bash


cd host_primer

nvcc -O3 -Xcompiler -fPIC --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -x cu -o  "VPrimer.o" "VPrimer.cu" -std=c++14

nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -link -o  "vprimer" ./VPrimer.o -std=c++14


cd ../host_probe

nvcc -O3 -Xcompiler -fPIC --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -x cu -o  "VPrimer.o" "VPrimer.cu" -std=c++14

nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -link -o  "vprimer" ./VPrimer.o -std=c++14


cd ../virus_primer

nvcc -O3 -Xcompiler -fPIC --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -x cu -o  "VPrimer.o" "VPrimer.cu" -std=c++14

nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -link -o  "vprimer" ./VPrimer.o -std=c++14

cd ../virus_probe

nvcc -O3 -Xcompiler -fPIC --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -x cu -o  "VPrimer.o" "VPrimer.cu" -std=c++14

nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -link -o  "vprimer" ./VPrimer.o -std=c++14

echo "build done."

