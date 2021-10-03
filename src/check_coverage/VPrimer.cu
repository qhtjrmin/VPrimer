
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "VPrimer.hpp"

int main(int argc, char *argv[]) {
	inputParameter* input = new inputParameter();
	double elapsedTime, st, et;

	if (readInputParameter(argc, argv, input) == false) {
		return -1;
	}

	initialization(input);

	st = getCurrentTime();
	variantBuilding();
	et = getCurrentTime();
	cout << "variantBuilding: " << (et - st) << endl;

	cout << "\n## organismBuilding" << endl;
	st = getCurrentTime();
	organismBuilding();
	et = getCurrentTime();
	cout << "organismBuilding: " << (et - st) << endl;

	cout << "\n## vidBuliding" << endl;
	runPthread(vidBuild, input->numOfThreads, &elapsedTime);
	cout << "Elapsed time (Stage 2) : " << elapsedTime << " sec"  << endl;

	cout << "\n## checkvId" << endl;
	st = getCurrentTime();
	checkvId();
	et = getCurrentTime();
	cout << "checkgId: " << (et - st) << endl;

	primerHCheck();

	runPthreadParam(wirteOutput, input->numOfThreads, 7, &elapsedTime);
	cout << "writeTime: " << elapsedTime << endl;

}
