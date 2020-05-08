#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdio.h>

#ifndef WIN64
#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/contrib/contrib.hpp>

#include "mex.h"
#include "matrix.h"
#include "main.cpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{


	if (nlhs != 1) {
			mexErrMsgIdAndTxt("MATLAB:outputmismatch",
												"Output arguments must be 1!");
	}

	if (nrhs != 2) {
			mexErrMsgIdAndTxt("MATLAB:inputmismatch",
												"Input arguments must be 2!");
	}

	if (!mxIsChar(prhs[0])){
		mexErrMsgIdAndTxt("MATLAB:inputmismatch",
												"The First Input arguments must be a string!");
	}

	if (! mxIsNumeric(prhs[1])){
		mexErrMsgIdAndTxt("MATLAB:inputmismatch",
												"The Second Input arguments must be a matrix!");
	}

	size_t n = mxGetN(prhs[1]);
	size_t m = mxGetM(prhs[1]);

	if (n != 3 || m != 3){
		mexErrMsgIdAndTxt("MATLAB:inputmismatch",
												"wrong matrix dimension!");
	}

	double *A = mxGetPr(prhs[1]);
	Eigen::Matrix3f K;
	for (size_t i=0; i < n*m; i++){
		K(i%n, i/m) = A[i];
	}


	//size_t buflen = mxGetN(prhs[0])*sizeof(mxChar)+1;
	char *dataFolder = mxArrayToString(prhs[0]);

   if (dataFolder == NULL)
     mexErrMsgTxt("Not enough heap space to hold converted string.");

 
 	//mxArray *assocFile;
 	//assocFile = mxCreateString("assocFile.txt");
 	//colorFile = mxCreateString("rgb.txt");

	//char *assocfile = mxArrayToString(assocFile);
	//char *colorfile = mxArrayToString(colorFile);

	std::vector<Eigen::Matrix4f> poses = ImageAlign(dataFolder, K);
	size_t numFrame = poses.size();
	size_t len = poses[0].size();

	std::cout << numFrame << std::endl;
	std::cout << len << std::endl;


	double *D;
	plhs[0] = mxCreateDoubleMatrix((mwSize)len, (mwSize)numFrame, mxREAL);
	D = mxGetPr(plhs[0]);

	for (int num =0 ; num< numFrame; num++){

		for (size_t j=0; j< poses[num].rows(); j++){
			for (size_t i=0; i<poses[num].cols();i++){
				*D++ = poses[num](j,i);
			}
		}		
		
	}
	
}