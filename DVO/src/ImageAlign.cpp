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


#include "dvo.hpp"
#include "tum_benchmark.hpp"

#define STR1(x)  #x
#define STR(x)  STR1(x)

Eigen::Matrix4f PoseEst(cv::Mat grayRef, cv::Mat depthRef, cv::Mat grayCur, cv::Mat depthCur, Eigen::Matrix3f K, Eigen::Matrix4f relPose)
{

//     Eigen::Matrix3f K;
// #if 1
//     // initialize intrinsic matrix: fr1
//     K <<    517.3, 0.0, 318.6,
//             0.0, 516.5, 255.3,
//             0.0, 0.0, 1.0;
    //dataFolder = "/rgbd_dataset_freiburg1_xyz/";
    //dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg1_desk2/";
// #else
//     dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg3_long_office_household/";
//     // initialize intrinsic matrix: fr3
//     K <<    535.4, 0.0, 320.1,
//             0.0, 539.2, 247.6,
//             0.0, 0.0, 1.0;
// #endif
    //std::cout << "Camera matrix: " << K << std::endl;

    // load file names

    //std::string dataFile_depth = dataFolder + "depth.txt";
    //std::string dataFile_color = dataFolder + "rgb.txt";

    //test
    
    // int numFrames = filesDepth.size();

    // int maxFrames = -1;
    // maxFrames = 3;

    // initialize
    
    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
   // std::vector<Eigen::Matrix4f> poses;
    //std::vector<double> timestamps;
    //poses.push_back(absPose);
    //timestamps.push_back(timestampsDepth[0]);

    //cv::Mat grayRef = loadGray(dataFolder + filesColor[0]);
    //cv::Mat depthRef = loadDepth(dataFolder + filesDepth[0]);
    int w = depthRef.cols;
    int h = depthRef.rows;

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<cv::Mat> grayRefPyramid;
    std::vector<cv::Mat> depthRefPyramid;
    dvo.buildPyramid(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    // process frames
   // double runtimeAvg = 0.0;
   // int framesProcessed = 0;
    //for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames); ++i)
   // {
        std::cout << "aligning frames " << std::endl;

        // load input frame
        //std::string fileColor1 = filesColor[i];
        //std::string fileDepth1 = filesDepth[i];
        //double timeDepth1 = timestampsDepth[i];
        //std::cout << "File " << i << ": " << fileColor1 << ", " << fileDepth1 << std::endl;
        //cv::Mat grayCur = loadGray(dataFolder + fileColor1);
        //cv::Mat depthCur = loadDepth(dataFolder + fileDepth1);
        // build pyramid
        std::vector<cv::Mat> grayCurPyramid;
        std::vector<cv::Mat> depthCurPyramid;
        dvo.buildPyramid(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

        // frame alignment
        double tmr = (double)cv::getTickCount();

        // Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        dvo.align(depthRefPyramid, grayRefPyramid, depthCurPyramid, grayCurPyramid, relPose);

        tmr = ((double)cv::getTickCount() - tmr)/cv::getTickFrequency();
        //runtimeAvg += tmr;

        // concatenate poses
        absPose = absPose * relPose.inverse();
        std::cout << "real Pose is " << std::endl;
        std::cout << relPose << std::endl;
        // //poses.push_back(absPose);
        //timestamps.push_back(timeDepth1);

        //depthRefPyramid = depthCurPyramid;
        //grayRefPyramid = grayCurPyramid;
        // std::cout << "pose for " << framesProcessed << " image: " << absPose << std::endl;

        //++framesProcessed;
   // }
    std::cout << "runtime: " << tmr * 1000.0 << " ms" << std::endl;

    // save poses
    //savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    //cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;

    return absPose;
}


int main(int argc, const char *argv[])
{
    // std::string dataFolder = std::string(STR(DVO_SOURCE_DIR)) + "/data/";
    // std::string dataFile_depth = dataFolder + "depth_test.txt";
    // std::string dataFile_color = dataFolder + "rgb_test.txt";
    std::string dataFolder = argv[1];
    std::string depthImg1 = dataFolder+argv[2];
    std::string depthImg2 = dataFolder+argv[3];
    std::string colorImg1 = dataFolder+argv[4];
    std::string colorImg2 = dataFolder+argv[5];
    

    Eigen::Matrix3f K;
    for (int i=0; i<9; i++) {
        K(i/3, i%3) = atof(argv[i+6]);
    }

    Eigen::Matrix4f relPose;
    for (int i=0; i<16; i++) {
        relPose(i/4, i%4) = atof(argv[i+14]);
    }

    cv::Mat grayRef = loadGray(colorImg1);
    cv::Mat depthRef = loadDepth(depthImg1);
    cv::Mat grayCur = loadGray(colorImg2);
    cv::Mat depthCur = loadDepth(depthImg2);

    Eigen::Matrix4f results;

    results = PoseEst(grayRef, depthRef, grayCur, depthCur, K, relPose);
    std::cout << "Aligned pose is " << results << std::endl;
       return 0;
}