# VPrimer
A method of designing and updating primer and probe with high variant coverage for RNA virus detection

# 1. Get VPrimer
To get VPrimer, follow the queries.
```
$ git clone https://github.com/qhtjrmin/VPrimer.git
```
The source codes are in "src" folder. Since the VPrimer consists of several steps, the vPrimer must be executed using a different input for each step.
The guidline for execution is in section 2.

## Environment
VPrimer needs the following softwares to run in the system:
- CUDA toolkit version 8 or higher.
- Nvidia driver (v384 or higher)
- gcc/g++ 4.8.x or later

#2. Running
## 2.1 Prepare input sequence data
The example of input files that can be used are in "input_test" folder. In the case of host data, it should be decompressed as the following command to use for input.
```
$ gunzip human_host.txt.gz
```
The required data is as follow:
- host sequence
- virus sequence
- mapping data for virus sequence (oid-vidset, sid-oid, sid-vidset)

The mapping data is for coverage checking and homology test against other virus.
Refer to test inputs and the paper for the format and meaning of the files. 

## 2.2 Input data generation
Refer to the following link:
https://github.com/Hajin-Jeon/VPrimer-input-generator

## 2.3 Get VPrimer execution file and run
The basic execution order is as follows.
- find host subsequences (host_primer and host_probe)
- find virus primer and probe candidates (virus_primer and virus_probe)
- variant coverage filtering: filter out sequences under 95% coverage (check_coverage)
- find primer pair having at least one proper probe (primer_pair)
- find final primer-probe sets (final_set)

You can run additional check_coverage, referring to the paper.
