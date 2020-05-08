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

std::vector<Eigen::Matrix4f> PoseEst(const std::string dataFolder, std::vector<std::string> &filesColor, std::vector<std::string> &filesDepth,  std::vector<double> &timestampsDepth,
    std::vector<double> &timestampsColor, Eigen::Matrix3f K, std::vector<Eigen::Matrix4f> &initial_poses)
{

//     Eigen::Matrix3f K;
// #if 1
//     // initialize intrinsic matrix: fr1
//     K <<    517.3, 0.0, 318.6,
//             0.0, 516.5, 255.3,
//             0.0, 0.0, 1.0;
    //dataFolder = "/rgbd_dataset_freiburg1_xyz/";
//     //dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg1_desk2/";
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
    
    int numFrames = filesDepth.size();

    //int maxFrames = -1;
    int maxFrames = 30;

    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity(); // for image 1
    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    cv::Mat grayRef = loadGray(dataFolder + filesColor[0]);
    cv::Mat depthRef = loadDepth(dataFolder + filesDepth[0]);
    int w = depthRef.cols;
    int h = depthRef.rows;

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<cv::Mat> grayRefPyramid;
    std::vector<cv::Mat> depthRefPyramid;
    dvo.buildPyramid(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    // process frames
    double runtimeAvg = 0.0;
    int framesProcessed = 0;
    for (size_t i = 1; (i < numFrames && i < maxFrames); ++i)
    {
        std::cout << "aligning frames " << (i-1) << " and " << i  << std::endl;

        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        double timeDepth1 = timestampsDepth[i];
        //std::cout << "File " << i << ": " << fileColor1 << ", " << fileDepth1 << std::endl;
        cv::Mat grayCur = loadGray(dataFolder + fileColor1);
        cv::Mat depthCur = loadDepth(dataFolder + fileDepth1);
        // build pyramid
        std::vector<cv::Mat> grayCurPyramid;
        std::vector<cv::Mat> depthCurPyramid;
        dvo.buildPyramid(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

        // frame alignment
        double tmr = (double)cv::getTickCount();

        //Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        //Eigen::Matrix4f relPose = initial_poses[i].inverse()*initial_poses[i-1];
        Eigen::Matrix4f relPose = initial_poses[i].inverse();
        if (relPose == Eigen::Matrix4f::Identity() & i > 0){
            relPose = absPose.inverse();
        }
        std::cout << "current initial pose is: " << std::endl;
        std::cout << relPose << std::endl;
        dvo.align(depthRefPyramid, grayRefPyramid, depthCurPyramid, grayCurPyramid, relPose);

        tmr = ((double)cv::getTickCount() - tmr)/cv::getTickFrequency();
        runtimeAvg += tmr;

        // concatenate poses
        //absPose = absPose * relPose.inverse();
        absPose = relPose.inverse();

        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

        //depthRefPyramid = depthCurPyramid;
        //grayRefPyramid = grayCurPyramid;
         
        // std::cout << "pose for " << framesProcessed << " image: " << absPose << std::endl;

        ++framesProcessed;
    }
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;

    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    //cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;

    return poses;
}

 std::vector<Eigen::Matrix4f> ImageAlign(const std::string dataFolder, Eigen::Matrix3f K)
{
    std::string assocFile = dataFolder + "assocFile.txt";
    std::string trajFile = dataFolder + "initial_traj.txt";

    std::vector<std::string> filesColor;
    std::vector<std::string> filesDepth;
    std::vector<double> timestampsDepth;
    std::vector<double> timestampsColor;
    std::vector<Eigen::Matrix4f> initial_poses;

    if (!loadAssoc(assocFile, filesDepth, filesColor, timestampsDepth, timestampsColor))
    {
        std::cout << "Assoc file could not be loaded!" << std::endl;
        //return NULL;
    }

    if (!loadPoses(trajFile, initial_poses))
    {
        std::cout << "Initialize poses as Identity matrix!" << std::endl;
        int numFrames = filesDepth.size();
        Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
        for (size_t i = 1; (i < (numFrames+1) ); ++i)
            {
                initial_poses.push_back(absPose);
            }


    }

    if (filesDepth.size()>initial_poses.size())
    {
        int numFrames = filesDepth.size();
        int numPoses = initial_poses.size();
        Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
        for (size_t i = 1; (i<( numFrames - numPoses + 1) ); ++i)
            {
                initial_poses.push_back(absPose);
            }
    }

    std::vector<Eigen::Matrix4f> results;
    results = PoseEst(dataFolder, filesColor, filesDepth, timestampsDepth, timestampsColor, K, initial_poses);
    //std::cout << results[1].size() << std::endl;

    return results;

}

int main(int argc, const char *argv[])
{
    
    std::string dataFolder = argv[1];
    Eigen::Matrix3f K;
    for (int i=0; i<9; i++) {
        K(i/3, i%3) = atof(argv[i+2]);
    }

    

    std::cout << dataFolder << std::endl;
    

    std::vector<Eigen::Matrix4f> results;
    results = ImageAlign(dataFolder, K);
       return 0;
}
