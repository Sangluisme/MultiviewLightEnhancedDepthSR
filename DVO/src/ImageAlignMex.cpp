#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

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
#include "ImageAlign.cpp"
#include "tum_benchmark.hpp"

// cv::Mat LoadGray(const std::string &filename)
// {
//     cv::Mat imgGray = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//     // convert gray to float
//     cv::Mat gray;
//     imgGray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);
//     return gray;
// }


// cv::Mat LoadDepth(const std::string &filename)
// {
//     //fill/read 16 bit depth image
//     cv::Mat imgDepthIn = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//     cv::Mat imgDepth;
//     imgDepthIn.convertTo(imgDepth, CV_32FC1, (1.0 / 5000.0));
//     return imgDepth;
// }


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{


	if (nlhs != 1) {
			mexErrMsgIdAndTxt("MATLAB:outputmismatch",
												"Output arguments must be 1!");
	}

	if (nrhs != 6) {
			mexErrMsgIdAndTxt("MATLAB:Inputmismatch",
												"Input arguments must be 6!");
	}

	
	char *depthImg1 = mxArrayToString(prhs[0]);

   if (depthImg1 == NULL)
     mexErrMsgTxt("Not enough heap space to hold converted string.");

 	char *depthImg2 = mxArrayToString(prhs[1]);

   if (depthImg2 == NULL)
     mexErrMsgTxt("Not enough heap space to hold converted string.");

 	
 	char *colorImg1 = mxArrayToString(prhs[2]);
   if (colorImg1 == NULL)
     mexErrMsgTxt("Not enough heap space to hold converted string.");


 	char *colorImg2 = mxArrayToString(prhs[3]);
   if (colorImg2 == NULL)
     mexErrMsgTxt("Not enough heap space to hold converted string.");

 	
 	cv::Mat grayRef = loadGray(colorImg1);
 	cv::Mat depthRef = loadDepth(depthImg1);
 	cv::Mat grayCur = loadGray(colorImg2);
 	cv::Mat depthCur = loadDepth(depthImg2);

  size_t n = mxGetN(prhs[4]);
  size_t m = mxGetM(prhs[4]);

  if (n != 3 || m != 3){
    mexErrMsgIdAndTxt("MATLAB:inputmismatch",
                        "wrong matrix dimension!");
  }

 	double *A = mxGetPr(prhs[4]);
	Eigen::Matrix3f K;
	for (size_t i=0; i < n*m; i++){
		K(i%n, i/m) = A[i];
	}

  size_t n_p = mxGetN(prhs[5]);
  size_t m_p = mxGetM(prhs[5]);

	if (n_p != 4 || m_p != 4){
		mexErrMsgIdAndTxt("MATLAB:inputmismatch",
                        "wrong initial pose dimension!");
  }

   	double *B = mxGetPr(prhs[5]);
	Eigen::Matrix4f relPose;
	for (size_t i=0; i < n_p*m_p; i++){
		relPose(i%n_p, i/m_p) = B[i];
	}

    // K <<    517.3, 0.0, 318.6,
    //         0.0, 516.5, 255.3,
    //         0.0, 0.0, 1.0;

	Eigen::Matrix4f poses = PoseEst(grayRef, depthRef, grayCur, depthCur, K, relPose);

	double *D;
	plhs[0] = mxCreateDoubleMatrix(4,4, mxREAL);
	D = mxGetPr(plhs[0]);
	size_t len = poses.size();
	for (size_t j=0; j< len; j++){
			D[j] = poses(j%4, j/4);
		}		
	
}