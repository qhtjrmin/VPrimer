/*
 * GPrimer.hpp
 *
 *  Created on: Feb 15, 2020
 *      Author: jmbae
 */

#ifndef GPRIMER_HPP_
#define GPRIMER_HPP_

#include "../lib/Constraints.hpp"
#include "../lib/FreeEnergyUtils.hpp"
#include "../lib/RunPthread.hpp"
#include "../lib/DataStructures.hpp"
#include "Input.hpp"

#include "sys/types.h"
#include "sys/sysinfo.h"

using namespace std;

#define DEBUG
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))



unsigned long usedMemory;

//-------------parameters for filtering---------------
//single filtering
int minLen; int maxLen;
double minGC; double maxGC;
double minTM; double maxTM;
int maxSC;
int endMaxSC;
int endMinDG;
int DGlen;
int maxHP;
int contiguous;

//pair filtering
int lenDiff; int TMDiff;
int minPS; int maxPS;
int maxPC; int endMaxPC;
//-----------------------------------------------------

//type definition
using Hmap_vec = unordered_map<char*, vector<char*>, hash<string>, CmpChar>;
using Hmap_set = unordered_map<char*, set<char*, set_Cmp>, hash<string>, CmpChar>;

//global memory
const int MAX_BUF_SIZE = 300;
int numOfThreads, numOfGPUs, numOfStreams;
vector<char*>* vecData;
const int sidLen = 7;
inputParameter* myInput;
int memGPU; //total memory of GPU

//hash map
unordered_map<char*, bool, hash<string>, CmpChar> primerH;
Hmap_set sidsetH;
Hmap_vec suffixH;
Hmap_vec seedHf, seedHr;
unordered_map<char*, unsigned int, hash<string>, CmpChar> seedHfIdx;
unordered_map<char*, unsigned int, hash<string>, CmpChar> seedHrIdx;
vector<pair<unsigned int, unsigned int>> sortedseedHfIdx;
vector<pair<unsigned int, unsigned int>> sortedseedHrIdx;

//mutex
pthread_mutex_t primerH_lock;
pthread_mutex_t sidsetH_lock;
pthread_mutex_t suffixH_lock;
pthread_mutex_t seedHf_lock;
pthread_mutex_t seedHr_lock;
pthread_mutex_t *seedHf_lock2;
pthread_mutex_t *seedHr_lock2;

//barrier
pthread_barrier_t barrThreads;
pthread_barrier_t barrGPUs;
pthread_barrier_t barr2GPUs;

//global for Step 4
bool probingEnd;
bool rank0End, rank1End;
vector<char>* P1f, *P1r;
vector<unsigned int>* SID1f, *SID1r;
vector<unsigned long>* P1foffset, *P1roffset;
bool* finCheck;
long fPcnt, rPcnt;
long f_threshold, r_threshold;
unsigned int* myWorkloadThresholdF, *myWorkloadThresholdR;
unsigned int* myseedHfcnt, *myseedHrcnt;
unsigned int* myFPcnt, *myRPcnt;
unsigned long* Gf_off, *Gr_off;
char** Gf_key, **Gr_key;
int workloadC1;
arraysForStep4** myArraysC3;
arraysForStep4** myArraysC3_dev;

//global for Step 5
double writeTime = 0;
float probingTime = 0;
long memSID;
unsigned int* sorted_fidx, *sorted_ridx;
unsigned int* sorted_rsid;
unsigned int* fsid, *rsid;
float **result;
unsigned int **Ooffset;
unsigned int *total_result3;
unsigned int *case3fcnt;
bool *finish;
int sid_workload;
int* fP_idx, *rP_idx;
int* sid_idx;
int* mySmallThreshold;
int* myLargeThreshold;
int smallThreshold = 10; //default decided by heuristic expeirments
int largeThreshold; //decided as distribution
pthread_mutex_t valid_lock;
long total_passed;
FILE** finalFile;
long total_input_workload;
int total_sid;
unsigned long total_input_line;
arraysForStep5 *tmpArraysStep5;
arraysForStep5 **myArraysStep5;

void initialization(inputParameter* input); //initialization for Step 2~4
void initialization2(long input_workload); //initialization for Step 5

void* stage1PrimerGeneration(void* rank){
	long myRank = (long) rank;
	long bufSize = 1024 * 5000;
	char buf[bufSize];
	char *sid = new char[100];
	char *sequence = new char[bufSize];
	char *ptr, *ptr2;
	char *Primer = new char[maxLen + 1];
	char *rPrimer = new char[maxLen + 2];
	bool check = true;
	int line = 0;
	char *fName = new char[100];
	string tmpfName;

	ifstream fin;
	fileReadOpen(&fin, myInput->inputPath, -1);

	FILE *fout;
	tmpfName = string(myInput->dirPath) + "/tmpC1.txt";
	strcpy(fName, tmpfName.c_str());
	fileWriteOpen(&fout, fName, myRank);

	while(fin.good()){
		fin.getline(buf, bufSize);

		if(fin.gcount() > 0){

			if (line % numOfThreads == myRank) {

				ptr = buf;
				if(strchr(ptr, '\t') != NULL){
					ptr2 = strchr(ptr, '\t');
					*ptr2 = '\0';
					sid = ptr;
					sequence = ++ptr2;

					if(stoi(sid) <= myInput->hostIdx){
					int sequenceLen = strlen(sequence);

					for (int i = 0; i < sequenceLen; i++) {
						//Primer, rPrimer initialization
						memset(Primer, 0, maxLen + 1);
						memset(rPrimer, 0, maxLen + 1);
						for (int k = minLen; k <= maxLen; k++) {
							if ((i + k) > sequenceLen) {
								break;
							}
							int t = 0;
							int r = k - 1;
							check = true;
							for (int j = 0; j < k; j++) {
								Primer[t] = sequence[i + j];
								if(sequence[i + j] != 'A' && sequence[i + j] != 'T' && sequence[i + j] != 'C' && sequence[i + j] != 'G')
									check = false;
/*								if (sequence[i + r] == 'A')
									rPrimer[t] = 'T';
								else if (sequence[i + r] == 'T')
									rPrimer[t] = 'A';
								else if (sequence[i + r] == 'C')
									rPrimer[t] = 'G';
								else if (sequence[i + r] == 'G')
									rPrimer[t] = 'C';
								else
									check = false;
*/								t++;
								r--;
							}
							if (check) {
								fprintf(fout, "%s\t%s\t%d\n", Primer, sid, i + 1);
//								fprintf(fout, "*%s\t%s\t%lu\n", rPrimer, sid,
//										i + strlen(rPrimer));
							}
						}
					}
					}
				}
			}
            line++;
		}
	}
	del(Primer); del(rPrimer);
	fin.close();
	fclose(fout);
	return NULL;
}

void stage1Sort(){
	string totalFile, sortedFile, command;
	ifstream fin;

	sortedFile = string(myInput->dirPath) + "/sorted.txt";

	//sort by primer
	command = "sort -k1,1 -k2,2n -S 50% --parallel "
			+ to_string(myInput->numOfThreads) + " -T " + string(myInput->dirPath) + " "
			+ string(myInput->dirPath) + "/tmpC1.txt_*" + " -o " + sortedFile;
	sysCall(command);

	sysCall("rm " + string(myInput->dirPath) + "/tmpC1.txt_*");
}

void stage1FileDistribution(){

	char buf[MAX_BUF_SIZE];
	int count = 0;
	long line = 0;
	char *primer, *etc;
	char *ptr, *ptr2;
	char *fName = new char[100];
	string tmpfName;
	char *before = new char[maxLen + 2];
	memset(before, 0, maxLen + 2);

	tmpfName = string(myInput->dirPath) + "/sorted.txt";
	strcpy(fName, tmpfName.c_str());

	ifstream fin;
	fileReadOpen(&fin, fName, -1);

	ifstream fin2;
	fileReadOpen(&fin2, fName, -1);
    long total_input = countLine(fin2);

	FILE **fout = new FILE *[numOfThreads];
	for(int i = 0; i < numOfThreads; i++)
		fileWriteOpen(&fout[i], myInput->c1Path, i);

    	while(fin.good()){

        fin.getline(buf,MAX_BUF_SIZE);

        if(fin.gcount() > 0){
            ptr = buf;
            if(strchr(ptr,'\t') != NULL){
				ptr2 = strchr(ptr, '\t');
				*ptr2 = '\0';
				primer = ptr;
				etc = ++ptr2;

				if (line >= (count + 1) * total_input / numOfThreads + 1) {
					if (strcmp(primer, before)) {
						fclose(fout[count]);
						count++;
					}
				}
				fprintf(fout[count], "%s\t%s\n", primer, etc);

				if (line >= ((count + 1) * total_input / numOfThreads))
					strcpy(before, primer);
				line++;
            }
        }
    }
    del(fout); del(fName); del(before);
}

void* stage2(void* rank) {
	long myRank = (long) rank;
	char buf[MAX_BUF_SIZE];
	char *untagPrimer;
	char *ptr, *ptr2;
	char **val;
	//bool pass = false; //check wheter primer is filtered or not
	long line = 0, count = 0;
	int cnt = 0;

	char* beforePrimer = new char[maxLen + 2];
	memset(beforePrimer, 0, maxLen + 2);

	char* beforeSid = new char[sidLen];
	memset(beforeSid, 0, sidLen);

	val = new char*[3];

	ifstream fin;
	fileReadOpen(&fin, myInput->c1Path, myRank);

	FILE *fout2;
//	fileWriteOpen(&fout1, myInput->c2Path, myRank); // C2.txt
	fileWriteOpen(&fout2, myInput->c1SidsetPath, myRank); //C1'.txt

	while (fin.good()) {

		fin.getline(buf, MAX_BUF_SIZE);

		if (fin.gcount() > 0) {

			//reading file
			ptr = buf;
			cnt = 0;
			while ((ptr2 = strchr(ptr, '\t')) != NULL) {
				*ptr2 = '\0';
				val[cnt] = ptr;
				ptr = ++ptr2;
				cnt++;
			}
			val[cnt] = ptr;

			if (val[0][0] == '*')
				untagPrimer = val[0] + 1;
			else
				untagPrimer = val[0];

#ifdef DEBUG
			if (myRank == 0) {
				if (line >= count * 10000000) {
					cout << beforePrimer << " " << val[0] << " " << untagPrimer
							<< " " << val[1] << " " << val[2] << " " << endl;
					count++;
				}
			}
#endif

			if (strcmp(val[0], beforePrimer)) { //if primer is changed (primer is sorted)

				if (line > 0)
					fprintf(fout2, "\n%s\t%s", val[0], val[1]); //write to C1'
				else
					fprintf(fout2, "%s\t%s", val[0], val[1]); //write to C1'
			}
			else {
				//before primer is the same as current primer but before primer is filterd out
				if (strcmp(beforeSid, val[1]))
					fprintf(fout2, "-%s", val[1]); //write to C1'
			}
		}
		strcpy(beforePrimer, val[0]);
		strcpy(beforeSid, val[1]);
		line++;
	}
	fprintf(fout2, "\n"); //write to C1'

	del(beforePrimer);
	del(beforeSid);
	del(val);

	fin.close();
//	fclose(fout1);
	fclose(fout2);

	return NULL;
}

void primerHCheck(){
	long numOfTrues = 0;
	long numOfFalses = 0;

	for(auto it = primerH.begin(); it != primerH.end(); it++){
		if((it->second) == true)
			numOfTrues++;
		else
			numOfFalses++;
	}
	cout << "primerH true: " << numOfTrues << " false: " << numOfFalses << "\n\n";
}

void* writeOutput(void *param) {
	paramThread *writeInput = (paramThread *) param;
	long myRank = writeInput->rank;
	int stage = writeInput->myParam;

	char *primer, *val;
	char *ptr, *ptr2;
	long line = 0, cnt = 0;
	char buf[MAX_BUF_SIZE];

	ifstream fin;
	fileReadOpen(&fin, myInput->c2Path, myRank);

	FILE* fout;
	if (stage == 3)
		fileWriteOpen(&fout, myInput->c3Path, myRank);
	else if (stage == 4)
		fileWriteOpen(&fout, myInput->c4Path1, myRank);
	else
		fileWriteOpen(&fout, myInput->c4Path2, myRank);

	while (fin.good()) {
		fin.getline(buf, MAX_BUF_SIZE);
		if (fin.gcount() > 0) {
			ptr = buf;
			ptr2 = strchr(ptr, '\t');
			*ptr2 = '\0';
			primer = ptr;
			val = ++ptr2;

#ifdef DEBUG
			if ((line >= cnt * 1000000) && (myRank == 0)) {
				cout << primer << " " << val << endl;
				cnt++;
			}
#endif

			if (primerH.find(primer) != primerH.end()) {
				if ((primerH.find(primer)->second) == true)
					fprintf(fout, "%s\t%s\n", primer, val);
			}
		}
		line++;
	}

	fin.close();
	fclose(fout);
	return NULL;
}
void* writeOutput2(void *param) {
	paramThread *writeInput = (paramThread *) param;
	long myRank = writeInput->rank;
	
	char *primer;
	char *ptr, *ptr2;
	long line = 0;
	char buf[MAX_BUF_SIZE];

	ifstream fin;
	fileReadOpen(&fin, myInput->myc1SidsetPath, myRank);

	FILE* fout;
	char *fName = new char[100]; string tmpfName;
	tmpfName = string(myInput->dirPath) + "/probe_sidset.txt";
	strcpy(fName, tmpfName.c_str());
	fileWriteOpen(&fout, fName, myRank);

	set<char*, set_Cmp> sidset;
	set<char*, set_Cmp>::iterator sid_it;

	while (fin.good()) {
		fin.getline(buf, MAX_BUF_SIZE);
		if (fin.gcount() > 0) {
			ptr = buf;
			if(strchr(ptr, '\t') != NULL){
				ptr2 = strchr(ptr, '\t');
				*ptr2 = '\0';
			}
			primer = ptr;
			//val = ++ptr2;
/*
#ifdef DEBUG
			if ((line >= cnt * 1000000) && (myRank == 0)) {
				cout << primer << " " << val << endl;
				cnt++;
			}
#endif
*/
			if (primerH.find(primer) != primerH.end()) {
				if ((primerH.find(primer)->second) == true){
					sidset = sidsetH.find(primer)->second;
					sid_it = sidset.begin();
					fprintf(fout, "%s", *sid_it);
					sid_it++;
					for(auto it = sid_it; it != (sidset.end()); it++){
						fprintf(fout, "-%s", (*it));
					}
					fprintf(fout, "\n");
					//fprintf(fout, "\t%s\t%s\n", primer, val);
				}
			}
		}
		line++;
	}

	fin.close();
	fclose(fout);
	return NULL;
}
void initialization(inputParameter* input) {

	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
    memGPU = prop.totalGlobalMem / (1024 * 1024);

	pthread_mutex_init(&primerH_lock, NULL);
	pthread_mutex_init(&sidsetH_lock, NULL);
	pthread_mutex_init(&seedHf_lock, NULL);
	pthread_mutex_init(&seedHr_lock, NULL);

	myInput = input;
	numOfThreads = input->numOfThreads;
	pthread_barrier_init(&barrThreads, 0, numOfThreads);
	numOfGPUs = input->numOfGPUs;
	pthread_barrier_init(&barrGPUs, 0, numOfGPUs);
	pthread_barrier_init(&barr2GPUs, 0, 2 * numOfGPUs);

	minLen = input->minLen;
	maxLen = input->maxLen;
	minGC = input->minGC;
	maxGC = input->maxGC;
	minTM = input->minTM;
	maxTM = input->maxTM;
	maxSC = input->maxSC;
	endMaxSC = input->endMaxSC;
	endMinDG = input->endMinDG;
	maxHP = input->maxHP;
	contiguous = input->contiguous;
	DGlen = 5;

	lenDiff = input->lenDiff;
	TMDiff = input->TMDiff;
	minPS = input->minPS;
	maxPS = input->maxPS;
	maxPC = input->maxPC;
	endMaxPC = input->endMaxPC;

	vecData = new vector<char*> [numOfThreads];
	myArraysC3 = new arraysForStep4* [numOfGPUs];
	myArraysC3_dev = new arraysForStep4* [numOfGPUs];
	for(int i = 0; i < numOfGPUs; i++){
		myArraysC3[i] = new arraysForStep4;
		myArraysC3_dev[i] = new arraysForStep4;
	}
	numOfStreams = 20;
}

void initialization2(long input_workload) {

	tmpArraysStep5 = new arraysForStep5;
	myArraysStep5 = new arraysForStep5 *[numOfGPUs];
	for(int i = 0; i < numOfGPUs; i++)
		myArraysStep5[i] = new arraysForStep5;

	sorted_fidx = new unsigned int[total_input_line];
	sorted_ridx = new unsigned int[total_input_line];

	fsid = new unsigned int[total_sid + 1];
	rsid = new unsigned int[total_sid + 1];
	sorted_rsid = new unsigned int[total_sid + 1];

	fP_idx = new int[numOfGPUs];
	rP_idx = new int[numOfGPUs];
	sid_idx = new int[numOfGPUs];
	mySmallThreshold = new int[numOfGPUs];
	myLargeThreshold = new int[numOfGPUs];

	result = new float*[numOfGPUs];
	Ooffset = new unsigned int*[numOfGPUs];

	tmpArraysStep5->FP = new unsigned char[maxLen * total_input_line];
	tmpArraysStep5->RP = new unsigned char[maxLen * total_input_line];
	tmpArraysStep5->Fpos = new unsigned int[total_input_line];
	tmpArraysStep5->Rpos = new unsigned int[total_input_line];
	tmpArraysStep5->Ftemp = new double[total_input_line];
	tmpArraysStep5->Fenergy = new double[total_input_line];
	tmpArraysStep5->Rtemp = new double[total_input_line];
	tmpArraysStep5->Renergy = new double[total_input_line];
	for (int i = 0; i < numOfGPUs; i++) {
		myArraysStep5[i]->FP = new unsigned char[input_workload * maxLen];
		myArraysStep5[i]->RP = new unsigned char[input_workload * maxLen];
		myArraysStep5[i]->Fpos = new unsigned int[input_workload];
		myArraysStep5[i]->Rpos = new unsigned int[input_workload];
		myArraysStep5[i]->FPoffset = new unsigned int[input_workload];
		myArraysStep5[i]->RPoffset = new unsigned int[input_workload];
		myArraysStep5[i]->Ftemp = new double[input_workload];
		myArraysStep5[i]->Rtemp = new double[input_workload];
		myArraysStep5[i]->Fenergy = new double[input_workload];
		myArraysStep5[i]->Renergy = new double[input_workload];
	}

	total_passed = 0;
}

#endif /* GPRIMER_HPP_ */
