
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

	et1 = getCurrentTime();

	cout << "Elapsed time(Stage 1) : " << (et1 - st1) << " sec" << endl << endl;



	cout << "\n## Stage 2 : single filtering" << endl;
	runPthread(stage2, input->numOfThreads, &elapsedTime);
	cout << "Elapsed time (Stage 2) : " << elapsedTime << " sec"  << endl << endl;

	cout << "## Stage 3 : 5' cross hybridization probing" << endl;
	runPthread(stage3, input->numOfThreads, &elapsedTime);
	stage3Delete();
	cout << "Elapsed time (Stage 3) : " << elapsedTime << " sec"  << endl << endl;

	if (input->isWriteOutput) {
		cout << "## Stage 3 (wirte) : 5' cross hybridization writing" << endl;
		runPthreadParam(wirteOutput, input->numOfThreads, 3, &elapsedTime);
		cout << "Elapsed time (Stage 3_write) : " << elapsedTime << " sec" << endl << endl;
	}

#ifdef DEBUG
	primerHCheck();
#endif

	st = getCurrentTime();

	for (int k = 1; k <= 2; k++) {

		cout << "## Stage 4_1 (k = " << k
				<< ") : general cross hybridization building" << endl;
		runPthreadParam(stage4Building, input->numOfThreads, k, &elapsedTime);
		cout << "Elapsed time (Stage 4_1, k = " << k << ") : " << elapsedTime
				<< " sec" << endl << endl;

		cout << "## Stage 4_2 (k = " << k
				<< ") : general cross hybridization preparing" << endl;
		runPthreadParam(stage4Prepare, input->numOfThreads, k, &elapsedTime);
		cout << "Elapsed time (Stage 4_2, k = " << k << ") : " << elapsedTime
				<< " sec"  << endl << endl;

		cout << "## Stage 4_3 (k = " << k
				<< ") : general cross hybridization probing" << endl;
		runPthreadParam(stage4Probing, input->numOfThreads, k, &elapsedTime);
		cout << "Elapsed time (Stage 4_3, k = " << k << ") : " << elapsedTime
				<< " sec"  << endl << endl;

		cout << "## Stage 4_4 (k = " << k
				<< ") : general cross hybridization updating" << endl;
		runPthread(stage4Update, input->numOfThreads, &elapsedTime);
		cout << "Elapsed time (Stage 4_4, k = " << k << ") : " << elapsedTime
				<< " sec" << endl << endl;

		if(input->isWriteOutput || k == 2){
			cout << "## Stage 4 (wirte) : general cross hybridization writing" << endl;
				runPthreadParam(wirteOutput, input->numOfThreads, 3 + k, &elapsedTime);
				cout << "Elapsed time (Stage 4_write) : " << elapsedTime << " sec"
						<< endl << endl;
		}
#ifdef DEBUG
		primerHCheck();
#endif
	}
	stage4Final();
	et = getCurrentTime();

	cout << "Elapsed time (Stage 4 total) : " << (et - st) << " sec" << endl << endl;

}
