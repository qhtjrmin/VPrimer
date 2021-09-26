
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "GPrimer.hpp"

int main(int argc, char *argv[]) {
	inputParameter* input = new inputParameter();
	double elapsedTime, st, et;
	double st1, et1;

	if (readInputParameter(argc, argv, input) == false) {
		return -1;
	}

	initialization(input);
	
	cout << "## Stage 1 : candidate generation" << endl;

	st1 = getCurrentTime();

	runPthread(stage1PrimerGeneration, input->numOfThreads, &elapsedTime);
	cout << "1_1(generation) : " << elapsedTime << endl;

	st = getCurrentTime();
	stage1Sort();
	et = getCurrentTime();
	cout << "1_2(sort) : " << (et - st) << endl;

	st = getCurrentTime();
	stage1FileDistribution();
	et = getCurrentTime();
	cout << "1_3(distribution) : " << (et - st) << endl;

	runPthread(stage2, input->numOfThreads, &elapsedTime);
	cout << "1_4(reorganization) : " << elapsedTime << " sec"  << endl;
	et1 = getCurrentTime();

	sysCall("rm " + string(myInput->dirPath) + "/C1.txt_*");
        sysCall("rm " + string(myInput->dirPath) + "/sorted.txt");

	cout << "Elapsed time(Stage 1) : " << (et1 - st1) << " sec" << endl << endl;

}
