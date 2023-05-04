/*
 * GPrimer.hpp
 *
 *  Created on: Feb 15, 2020
 *      Author: jmbae
 */

#ifndef GPRIMER_HPP_
#define GPRIMER_HPP_

#include "Input.hpp"

#include "../lib/Constraints.hpp"
#include "../lib/FreeEnergyUtils.hpp"
#include "../lib/RunPthread.hpp"
#include "../lib/Step5Header.hpp"
#include "../lib/KernelFunction.hpp"

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
unordered_map<char*, set<int>, hash<string>, CmpChar>  sidsetH;
Hmap_vec suffixH;
Hmap_vec seedHf, seedHr;
unordered_map<char*, unsigned int, hash<string>, CmpChar> seedHfIdx;
unordered_map<char*, unsigned int, hash<string>, CmpChar> seedHrIdx;
vector<pair<unsigned int, unsigned int>> sortedseedHfIdx;
vector<pair<unsigned int, unsigned int>> sortedseedHrIdx;
unordered_map<char*, set<int>, hash<string>, CmpChar> sid2vidH;
unordered_map<char*, set<int>, hash<string>, CmpChar> vidsetH;
vector<pair<int,int>> organismV;

//mutex
pthread_mutex_t primerH_lock;
pthread_mutex_t sidsetH_lock;
pthread_mutex_t suffixH_lock;
pthread_mutex_t seedHf_lock;
pthread_mutex_t seedHr_lock;
pthread_mutex_t *seedHf_lock2;
pthread_mutex_t *seedHr_lock2;
pthread_mutex_t vidsetH_lock;

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
vector<unsigned int> sorted_fsid;
char *finName, *foutName;
arraysForStep5 *tmpArraysStep5;
arraysForStep5 **myArraysStep5;

void initialization(inputParameter* input); //initialization for Step 2~4
void initialization2(long input_workload); //initialization for Step 5

void organismBuilding(){//seperated by '-'. build organismV

	long bufSize = 1024 * 5000;
	char buf[bufSize];
	char *ptr, *ptr2;

	ifstream fin;
	fileReadOpen(&fin, myInput->oIdPath, -1);

	char *vidset = new char[bufSize];

	int oid = 0;
	int svid, evid;

	while(fin.good()){
		fin.getline(buf, bufSize);

		if(fin.gcount() > 0){
			ptr = buf;
			if(strchr(ptr, '\t') != NULL){
				ptr2 = strpbrk(ptr, " \t");
				*ptr2 = '\0';
				strcpy(vidset, ++ptr2);

				ptr = vidset;
			
				if(strchr(ptr, '-') != NULL){				
					ptr2 = strchr(ptr, '-');
					*ptr2 = '\0';
					svid = stoi(ptr);
				
					ptr = ++ptr2;
					while((ptr2 = strchr(ptr, '-')) != NULL){
						*ptr2 = '\0';
						ptr = ++ptr2;
					}
					evid = stoi(ptr);
				}
				else{
					svid = stoi(vidset);
					evid = stoi(vidset);
				}
				organismV.push_back(pair<int,int>(svid, evid));

				oid++;
			}
		}
	}
	/*	check organismV
	for (int i = 0; i < organismV.size(); i++) {
    cout << "organismV[" << i << "]: " << organismV[i].first << "-" << organismV[i].second << endl;
}*/


}

void variantBuilding(){//separated as sid. build sid2vidH

	long bufSize = 1024 * 5000;
	char buf[bufSize];
	char *ptr, *ptr2;

	ifstream fin;
	fileReadOpen(&fin, myInput->vIdPath, -1);

	char *vid = new char[sidLen + 1];
	char *vidset = new char[bufSize];
	char *sid;

	while(fin.good()){
		fin.getline(buf, bufSize);

		if(fin.gcount() > 0){
			ptr = buf;
			ptr2 = strpbrk(ptr, " \t");
			*ptr2 = '\0';

			sid = new char[sidLen + 1];
			strcpy(sid, ptr);
			strcpy(vidset, ++ptr2);
			ptr = vidset;

			set<int> tmp;
			while((ptr2 = strchr(ptr, '-')) != NULL){
				*ptr2 = '\0';
				tmp.insert(stoi(ptr));
				ptr = ++ptr2;
			}
			tmp.insert(stoi(ptr));

			sid2vidH.insert(pair<char*, set<int>>(sid, tmp));
		}
	}
	/* check sid2vidH
	for (auto it = sid2vidH.begin(); it != sid2vidH.end(); ++it) {
    cout << "sid: " << it->first << endl;
    for (auto vid : it->second) {
        cout << "vid: " << vid << endl;
    }
}
*/


}

void* vidBuild(void* rank) {
	long myRank = (long) rank;
	char buf[MAX_BUF_SIZE];
	char *untagPrimer;
	char *primer;
	char *ptr, *ptr2;
	char **val;
	long line = 0, count = 0;
	int cnt = 0;

	char* beforePrimer = new char[maxLen + 2];
	memset(beforePrimer, 0, maxLen + 2);

	val = new char*[3];

	ifstream fin;
	fileReadOpen(&fin, myInput->inputPath, myRank);
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

				if(stoi(val[1]) > myInput->hostIdx){					
						
						primer = new char[maxLen + 2];
						strcpy(primer, val[0]);

						//primerH update
						pthread_mutex_lock(&primerH_lock);
                        primerH.insert(pair<char*, bool>(primer, true));
                        pthread_mutex_unlock(&primerH_lock);

						//sidsetH update
						set<int> tmp2;
						tmp2.insert(stoi(val[1]));
						pthread_mutex_lock(&sidsetH_lock);
						if(sidsetH.find(primer) == sidsetH.end()){
							sidsetH.insert(pair<char*, set<int>>(primer, tmp2));
						}
						pthread_mutex_unlock(&sidsetH_lock);
						freeContainer(tmp2);
						

						//vidsetH update
						set<int> tmp;
						tmp = sid2vidH.find(val[1])->second;
						pthread_mutex_lock (&vidsetH_lock);
						if (vidsetH.find(primer) == vidsetH.end())
							vidsetH.insert(pair<char*, set<int>>(primer, tmp));
						pthread_mutex_unlock(&vidsetH_lock);
						freeContainer(tmp);
				}
			}

			else{
				//in the case of (beforePrimer != Primer)
				pthread_mutex_lock (&vidsetH_lock);
				if (vidsetH.find(val[0]) != vidsetH.end()) {
					for(auto it = (sid2vidH.find(val[1])->second).begin(); it != (sid2vidH.find(val[1])->second).end(); it++)
						(vidsetH.find(val[0])->second).insert((*it));
				}
				pthread_mutex_unlock(&vidsetH_lock);

				pthread_mutex_lock(&sidsetH_lock);
				if(sidsetH.find(val[0]) != sidsetH.end()){
					(sidsetH.find(val[0])->second).insert(stoi(val[1]));
				}
				pthread_mutex_unlock(&sidsetH_lock);
			}
		}
		strcpy(beforePrimer, val[0]);
		line++;
	}

	del(beforePrimer);
	del(val);

	fin.close();
	return NULL;
}

void checkvId(){
	char* primer;
	unsigned int svid, evid;

	cout << "vidsetH size: " << vidsetH.size() << endl;

	long total_organism = fileReadingLine(myInput->oIdPath);
	set<int> vidset;
	bool check;
	int passed_organism;
	unsigned int totalvid;
	unsigned int numOfpassed = 0;
	vector<int> passed_oidset;
	for(auto it = vidsetH.begin(); it != vidsetH.end(); it++){

		primer = it->first;
		vidset = it->second;

		passed_oidset.clear();

		for(int i = 0; i < total_organism; i++){

			svid = organismV[i].first;
			evid = organismV[i].second;
			totalvid = evid - svid + 1;
			check = true;

			numOfpassed = 0;

			for(int tmpIdx = svid; tmpIdx <= evid; tmpIdx++){
				if(vidset.find(tmpIdx) != vidset.end()){
					numOfpassed++;
				}
			}

			if(numOfpassed >= totalvid * 0.95)
				check = true;
			else
				check = false;

			passed_organism = i + 1;

			if(check){
				passed_oidset.push_back(passed_organism);
				i = total_organism;
			}
		}
		if(passed_oidset.size() == 0){
			pthread_mutex_lock(&primerH_lock);
			primerH.find(primer)->second = false;
			pthread_mutex_unlock(&primerH_lock);
		}
	}
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

void* wirteOutput(void *param) {
	paramThread *writeInput = (paramThread *) param;
	long myRank = writeInput->rank;
	int stage = writeInput->myParam;

	char *primer, *val;
	char *ptr, *ptr2;
	long line = 0, cnt = 0;
	char buf[MAX_BUF_SIZE];

	ifstream fin;
	fileReadOpen(&fin, myInput->inputPath, myRank);

	FILE* fout;
	if (stage == 3)
		fileWriteOpen(&fout, myInput->c3Path, myRank);
	else if (stage == 4)
		fileWriteOpen(&fout, myInput->c4Path1, myRank);
	else if (stage == 7)
		fileWriteOpen(&fout, myInput->outputPath, myRank);
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

void initialization(inputParameter* input) {

	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop, 0);
    memGPU = prop.totalGlobalMem / (1024 * 1024);

	pthread_mutex_init(&primerH_lock, NULL);
	pthread_mutex_init(&sidsetH_lock, NULL);
	pthread_mutex_init(&seedHf_lock, NULL);
	pthread_mutex_init(&seedHr_lock, NULL);
	pthread_mutex_init(&vidsetH_lock, NULL);

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
