#!/bin/bash

# Usage
function usage {
  echo "Usage: $0 -H <host_DB> -V <virus_DB> -o <oid_vidset> -v <sid_vidset> [-t <cpu_threads>] [-g <gpu_threads>]" >&2
  exit 1
}

while getopts ":H:V:o:v:t:g:" opt; do
  case $opt in
    H ) host_path="$OPTARG";;
    V ) virus_path="$OPTARG";;
    o ) organism_path="$OPTARG";;
    v ) sid_path="$OPTARG";;
    t ) cpu_threads="$OPTARG";;
    g ) gpu_threads="$OPTARG";;
    \? ) echo "Invalid option: -$OPTARG" >&2
         usage;;
    : ) echo "Option -$OPTARG requires an argument." >&2
         usage;;
    * ) usage;;
  esac
done

# check necessary input
if [ -z "$host_path" ] || [ -z "$virus_path" ] || [ -z "$organism_path" ] || [ -z "$sid_path" ]; then
  echo "Error: Required options are missing." >&2
  usage
fi

# print value of each option
echo "Host path: $host_path"
echo "Virus path: $virus_path"
echo "Organism-vid mapping path: $organism_path"
echo "Sid-vid mapping path: $sid_path"
echo "Number of CPU threads: ${cpu_threads:-20}"
echo "Number of GPU threads: ${gpu_threads:-1}"

host_size=$(wc -l < "$host_path")

# make folders
mkdir -p host_primer/directory
mkdir -p host_primer/output
mkdir -p host_probe/directory
mkdir -p host_probe/output
mkdir -p virus_primer/directory
mkdir -p virus_primer/output
mkdir -p virus_probe/directory
mkdir -p virus_probe/output
mkdir -p check_coverage/directory
mkdir -p check_coverage/output

# run hosts
echo "[start VPrimer]"
echo "[host primer and probe]"
./host_primer/vprimer -i $host_path -d ./host_primer/directory/ -o ./host_primer/output/ -s $host_size -t $cpu_threads -g $gpu_threads
./host_probe/vprimer -i $host_path -d ./host_probe/directory/ -o ./host_probe/output/ -s $host_size -t $cpu_threads -g $gpu_threads

# run viruses
echo "[virus primer and probe]"
./virus_primer/vprimer -i $virus_path -d ./virus_primer/directory/ -o ./virus_primer/output/ -s $host_size -t $cpu_threads -g $gpu_threads
./virus_probe/vprimer -i $virus_path -d ./virus_probe/directory/ -o ./virus_probe/output/ -s $host_size -t $cpu_threads -g $gpu_threads

# check coverage
echo "[check coverage]"
./check_coverage/vprimer -og $organism_path -v $sid_path -i virus_primer/directory/C4_2.txt -d check_coverage/directory/ -o check_coverage/output/result -s $host_size -t $cpu_threads -g $gpu_threads

sort -k2,2n -k3,3n -k1,1 -S 80% --parallel $cpu_threads -T check_coverage/directory/ check_coverage/output/result_* -o check_coverage/output/output
rm check_coverage/output/result_*

echo "[job done.]"
