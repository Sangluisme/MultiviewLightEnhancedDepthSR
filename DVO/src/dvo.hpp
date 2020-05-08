// Copyright 2016 Robert Maier, Technical University Munich
#ifndef DVO_H
#define DVO_H

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


class DVO
{
public:
    enum MinimizationAlgo
    {
        GaussNewton = 0,
        GradientDescent = 1,
        LevenbergMarquardt = 2
    };

    DVO();
    ~DVO();

    void init(int w, int h, const Eigen::Matrix3f &K);

    void buildPyramid(const cv::Mat &depth, const cv::Mat &gray, std::vector<cv::Mat> &depthPyramid, std::vector<cv::Mat> &grayPyramid);

    void align(const cv::Mat &depthRef, const cv::Mat &grayRef,
               const cv::Mat &depthCur, const cv::Mat &grayCur,
               Eigen::Matrix4f &pose);

    void align(const std::vector<cv::Mat> &depthRefPyramid, const std::vector<cv::Mat> &grayRefPyramid,
               const std::vector<cv::Mat> &depthCurPyramid, const std::vector<cv::Mat> &grayCurPyramid,
               Eigen::Matrix4f &pose);


private:
    cv::Mat downsampleGray(const cv::Mat &gray);
    cv::Mat downsampleDepth(const cv::Mat &depth);

    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t);
    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose);
    void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi);

    void computeGradient(const cv::Mat &gray, cv::Mat &gradient, int direction);
    float interpolate(const float* ptrImgIntensity, float x, float y, int w, int h);
    float calculateError(const float* residuals, int n);
    void calculateErrorImage(const float* residuals, int w, int h, cv::Mat &errorImage);
    void calculateError(const cv::Mat &grayRef, const cv::Mat &depthRef,
                        const cv::Mat &grayCur, const cv::Mat &depthCur,
                        const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                        float* residuals);
    void convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi);
    
    void calculateMeanStdDev(const float* residuals, float &mean, float &stdDev, int n);
    void computeWeights(const float* residuals, float* weights, int n);
    void applyWeights(const float* weights, float* residuals, int n);

    void deriveNumeric(const cv::Mat &grayRef, const cv::Mat &depthRef,
                                      const cv::Mat &grayCur, const cv::Mat &depthCur,
                                      const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                                      float* residuals, float* J);
    void deriveAnalytic(const cv::Mat &grayRef, const cv::Mat &depthRef,
                       const cv::Mat &grayCur, const cv::Mat &depthCur,
                       const cv::Mat &gradX_, const cv::Mat &gradY_,
                       const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                       float* residuals, float* J);

    void compute_JtR(const float* J, const float* residuals, Vec6f &b, int validRows);
    void compute_JtJ(const float* J, Mat6f &A, const float* weights, int validRows, bool useWeights);

    int numPyramidLevels_;
    std::vector<Eigen::Matrix3f> kPyramid_;
    std::vector<cv::Size> sizePyramid_;
    bool useWeights_;
    int numIterations_;

    std::vector<cv::Mat> gradX_;
    std::vector<cv::Mat> gradY_;
    std::vector<float*> J_;
    std::vector<float*> residuals_;
    std::vector<float*> weights_;

    MinimizationAlgo algo_;
};

#endif
