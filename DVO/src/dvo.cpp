// Copyright 2016 Robert Maier, Technical University Munich
#include "dvo.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


DVO::DVO() :
    numPyramidLevels_(2),
    useWeights_(true),
    numIterations_(500),
    algo_(LevenbergMarquardt)
    //algo_(GaussNewton)
{
}


DVO::~DVO()
{
    for (int i = 0; i < numPyramidLevels_; ++i)
    {
        delete[] J_[i];
        delete[] residuals_[i];
        delete[] weights_[i];
    }
}


void DVO::init(int w, int h, const Eigen::Matrix3f &K)
{
    // pyramid level size
    int wDown = w;
    int hDown = h;
    int n = wDown*hDown;
    sizePyramid_.push_back(cv::Size(wDown, hDown));

    // gradients
    cv::Mat gradX = cv::Mat::zeros(h, w, CV_32FC1);
    gradX_.push_back(gradX);
    cv::Mat gradY = cv::Mat::zeros(h, w, CV_32FC1);
    gradY_.push_back(gradY);

    // Jacobian
    float* J = new float[n*6];
    J_.push_back(J);
    // residuals
    float* residuals = new float[n];
    residuals_.push_back(residuals);
    // per-residual weights
    float* weights = new float[n];
    weights_.push_back(weights);

    // camera matrix
    kPyramid_.push_back(K);

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        // pyramid level size
        wDown = wDown / 2;
        hDown = hDown / 2;
        int n = wDown*hDown;
        sizePyramid_.push_back(cv::Size(wDown, hDown));

        // gradients
        cv::Mat gradXdown = cv::Mat::zeros(hDown, wDown, CV_32FC1);
        gradX_.push_back(gradXdown);
        cv::Mat gradYdown = cv::Mat::zeros(hDown, wDown, CV_32FC1);
        gradY_.push_back(gradYdown);

        // Jacobian
        float* J = new float[n*6];
        J_.push_back(J);
        // residuals
        float* residuals = new float[n];
        residuals_.push_back(residuals);
        // per-residual weights
        float* weights = new float[n];
        weights_.push_back(weights);

        // downsample camera matrix
        Eigen::Matrix3f kDown = kPyramid_[i-1];
        kDown(0, 2) += 0.5f;
        kDown(1, 2) += 0.5f;
        kDown.topLeftCorner(2, 3) = kDown.topLeftCorner(2, 3) * 0.5f;
        kDown(0, 2) -= 0.5f;
        kDown(1, 2) -= 0.5f;
        kPyramid_.push_back(kDown);
        //std::cout << "Camera matrix (level " << i << "): " << kDown << std::endl;
    }
}


void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t)
{
    // rotation
    Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
    Eigen::Matrix4f mat = se3.matrix();
    rot = mat.topLeftCorner(3, 3);
    t = mat.topRightCorner(3, 1);
}


void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose)
{
    Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
    pose = se3.matrix();
}


void DVO::convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi)
{
    Sophus::SE3f se3(rot, t);
    xi = Sophus::SE3f::log(se3);
}


void DVO::convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi)
{
    Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
    Eigen::Vector3f t = pose.topRightCorner(3, 1);
    convertTfToSE3(rot, t, xi);
}


cv::Mat DVO::downsampleGray(const cv::Mat &gray)
{
    const float* ptrIn = (const float*)gray.data;
    int w = gray.cols;
    int h = gray.rows;
    int wDown = w/2;
    int hDown = h/2;

    cv::Mat grayDown = cv::Mat::zeros(hDown, wDown, gray.type());
    float* ptrOut = (float*)grayDown.data;
    for (size_t y = 0; y < hDown; ++y)
    {
        for (size_t x = 0; x < wDown; ++x)
        {
            float sum = 0.0f;
            sum += ptrIn[2*y * w + 2*x] * 0.25f;
            sum += ptrIn[2*y * w + 2*x+1] * 0.25f;
            sum += ptrIn[(2*y+1) * w + 2*x] * 0.25f;
            sum += ptrIn[(2*y+1) * w + 2*x+1] * 0.25f;
            ptrOut[y*wDown + x] = sum;
        }
    }

    return grayDown;
}


cv::Mat DVO::downsampleDepth(const cv::Mat &depth)
{
    const float* ptrIn = (const float*)depth.data;
    int w = depth.cols;
    int h = depth.rows;
    int wDown = w/2;
    int hDown = h/2;

    // downscaling by averaging the inverse depth
    cv::Mat depthDown = cv::Mat::zeros(hDown, wDown, depth.type());
    float* ptrOut = (float*)depthDown.data;
    for (size_t y = 0; y < hDown; ++y)
    {
        for (size_t x = 0; x < wDown; ++x)
        {
            float d0 = ptrIn[2*y * w + 2*x];
            float d1 = ptrIn[2*y * w + 2*x+1];
            float d2 = ptrIn[(2*y+1) * w + 2*x];
            float d3 = ptrIn[(2*y+1) * w + 2*x+1];

            int cnt = 0;
            float sum = 0.0f;
            if (d0 != 0.0f)
            {
                sum += 1.0f / d0;
                ++cnt;
            }
            if (d1 != 0.0f)
            {
                sum += 1.0f / d1;
                ++cnt;
            }
            if (d2 != 0.0f)
            {
                sum += 1.0f / d2;
                ++cnt;
            }
            if (d3 != 0.0f)
            {
                sum += 1.0f / d3;
                ++cnt;
            }

            if (cnt > 0)
            {
                float dInv = sum / float(cnt);
                if (dInv != 0.0f)
                    ptrOut[y*wDown + x] = 1.0f / dInv;
            }
        }
    }

    return depthDown;
}


void DVO::computeGradient(const cv::Mat &gray, cv::Mat &gradient, int direction)
{
    int dirX = 1;
    int dirY = 0;
    if (direction == 1)
    {
        dirX = 0;
        dirY = 1;
    }

    // compute gradient manually using finite differences
    int w = gray.cols;
    int h = gray.rows;
    const float* ptrIn = (const float*)gray.data;
    gradient.setTo(0);
    float* ptrOut = (float*)gradient.data;

    int yStart = dirY;
    int yEnd = h - dirY;
    int xStart = dirX;
    int xEnd = w - dirX;
    for (size_t y = yStart; y < yEnd; ++y)
    {
        for (size_t x = xStart; x < xEnd; ++x)
        {
            float v0;
            float v1;
            if (direction == 1)
            {
                // y-direction
                v0 = ptrIn[(y-1)*w + x];
                v1 = ptrIn[(y+1)*w + x];
            }
            else
            {
                // x-direction
                v0 = ptrIn[y*w + (x-1)];
                v1 = ptrIn[y*w + (x+1)];
            }
            ptrOut[y*w + x] = 0.5f * (v1 - v0);
        }
    }
}


float DVO::interpolate(const float* ptrImgIntensity, float x, float y, int w, int h)
{
    float valCur = std::numeric_limits<float>::quiet_NaN();

#if 0
    // direct lookup, no interpolation
    int x0 = static_cast<int>(x + 0.5f);
    int y0 = static_cast<int>(y + 0.5f);
    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
        valCur = ptrImgIntensity[y0*w + x0];
#else
    //bilinear interpolation
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float x1_weight = x - static_cast<float>(x0);
    float y1_weight = y - static_cast<float>(y0);
    float x0_weight = 1.0f - x1_weight;
    float y0_weight = 1.0f - y1_weight;

    if (x0 < 0 || x0 >= w)
        x0_weight = 0.0f;
    if (x1 < 0 || x1 >= w)
        x1_weight = 0.0f;
    if (y0 < 0 || y0 >= h)
        y0_weight = 0.0f;
    if (y1 < 0 || y1 >= h)
        y1_weight = 0.0f;
    float w00 = x0_weight * y0_weight;
    float w10 = x1_weight * y0_weight;
    float w01 = x0_weight * y1_weight;
    float w11 = x1_weight * y1_weight;

    float sumWeights = w00 + w10 + w01 + w11;
    float sum = 0.0f;
    if (w00 > 0.0f)
        sum += ptrImgIntensity[y0*w + x0] * w00;
    if (w01 > 0.0f)
        sum += ptrImgIntensity[y1*w + x0] * w01;
    if (w10 > 0.0f)
        sum += ptrImgIntensity[y0*w + x1] * w10;
    if (w11 > 0.0f)
        sum += ptrImgIntensity[y1*w + x1] * w11;

    if (sumWeights > 0.0f)
        valCur = sum / sumWeights;
#endif

    return valCur;
}


float DVO::calculateError(const float* residuals, int n)
{
    float error = 0.0f;
    int numValid = 0;
    for (int i = 0; i < n; ++i)
    {
        if (residuals[i] != 0.0f)
        {
            error += residuals[i] * residuals[i];
            ++numValid;
        }
    }
    if (numValid > 0)
        error = error / static_cast<float>(numValid);
    return error;
}


void DVO::calculateErrorImage(const float* residuals, int w, int h, cv::Mat &errorImage)
{
    cv::Mat imgResiduals = cv::Mat::zeros(h, w, CV_32FC1);
    float* ptrResiduals = (float*)imgResiduals.data;

    // fill residuals image
    for (size_t y = 0; y < h; ++y)
    {
        for (size_t x = 0; x < w; ++x)
        {
            size_t off = y*w + x;
            if (residuals[off] != 0.0f)
                ptrResiduals[off] = residuals[off];
        }
    }

    imgResiduals.convertTo(errorImage, CV_8SC1, 127.0);
}


void DVO::calculateError(const cv::Mat &grayRef, const cv::Mat &depthRef,
                         const cv::Mat &grayCur, const cv::Mat &depthCur,
                         const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                         float* residuals)
{
    // create residual image
    int w = grayRef.cols;
    int h = grayRef.rows;

    // camera intrinsics
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;

    // convert SE3 to rotation matrix and translation vector
    Eigen::Matrix3f rotMat;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rotMat, t);

    const float* ptrGrayRef = (const float*)grayRef.data;
    const float* ptrDepthRef = (const float*)depthRef.data;
    const float* ptrGrayCur = (const float*)grayCur.data;
    const float* ptrDepthCur = (const float*)depthCur.data;

    for (size_t y = 0; y < h; ++y)
    {
        for (size_t x = 0; x < w; ++x)
        {
            size_t off = y*w + x;
            float residual = 0.0f;

            // project 2d point back into 3d using its depth
            float dRef = ptrDepthRef[y*w + x];
            if (dRef > 0.0)
            {
                float x0 = (static_cast<float>(x) - cx) * fxInv;
                float y0 = (static_cast<float>(y) - cy) * fyInv;
                float scale = 1.0f;
                //scale = std::sqrt(x0*x0 + y0*y0 + 1.0f);
                dRef = dRef * scale;
                x0 = x0 * dRef;
                y0 = y0 * dRef;

                // transform reference 3d point into current frame
                // reference 3d point
                Eigen::Vector3f pt3Ref(x0, y0, dRef);
                Eigen::Vector3f pt3Cur = rotMat * pt3Ref + t;
                if (pt3Cur[2] > 0.0f)
                {
                    // project 3d point to 2d
                    Eigen::Vector3f pt2CurH = K * pt3Cur;
                    float ptZinv = 1.0f / pt2CurH[2];
                    float px = pt2CurH[0] * ptZinv;
                    float py = pt2CurH[1] * ptZinv;

                    // interpolate residual
                    float valCur = interpolate(ptrGrayCur, px, py, w, h);
                    if (!std::isnan(valCur))
                    {
                        float valRef = ptrGrayRef[off];
                        float valDiff = valRef - valCur;
                        residual = valDiff;
                    }
                }
            }
            residuals[off] = residual;
        }
    }
}


void DVO::calculateMeanStdDev(const float* residuals, float &mean, float &stdDev, int n)
{
    float meanVal = 0.0f;
    for (int i = 0; i < n; ++i)
        meanVal += residuals[i];
    mean = meanVal / static_cast<float>(n);

    float variance = 0.0f;
    for (int i = 0; i < n; ++i)
        variance += (residuals[i] - mean) * (residuals[i] - mean);
    stdDev = std::sqrt(variance);
}


void DVO::computeWeights(const float* residuals, float* weights, int n)
{
#if 0
    // no weighting
    for (int i = 0; i < n; ++i)
        weights[i] = 1.0f;
#if 0
    // squared residuals
    for (int i = 0; i < n; ++i)
        residuals[i] = residuals[i] * residuals[i];
    return;
#endif
#endif

    // compute mean and standard deviation
    float mean, stdDev;
    calculateMeanStdDev(residuals, mean, stdDev, n);

    // compute robust Huber weights
    
    float k = 1.345f * stdDev;
    //float k = 0.15f * stdDev;
    for (int i = 0; i < n; ++i)
    {
        float w;
        if (std::abs(residuals[i]) <= k)
            w = 1.0f;
        else
            w = k / std::abs(residuals[i]);
        weights[i] = w;
    }
    /*
    #if 0
    // compute Tukey weights
    float k = 4.6851f * stdDev;
    for (int i = 0; i<n; i++)
    {
        float w;
        if (std::abs(residuals[i]) <= k)
            w = std::pow((1.0f-residuals[i]*residuals[i] / (k*k)),2);
        else
            w = 0.0f;
        weights[i] = w;
    }
    
    // compute Cauchy weights
    float k = 1.35f * stdDev;
    for (int i = 0; i<n; i++)
    {
        float w;       
        w = 1.0f/(1+(residuals[i]*residuals[i] / (k*k)));        
        weights[i] = w;
    }
    */
    
}


void DVO::applyWeights(const float* weights, float* residuals, int n)
{
    for (size_t i = 0; i < n; ++i)
    {
        // weight residual
        residuals[i] = residuals[i] * weights[i];
    }
}


void DVO::deriveNumeric(const cv::Mat &grayRef, const cv::Mat &depthRef,
                                  const cv::Mat &grayCur, const cv::Mat &depthCur,
                                  const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                                  float* residuals, float* J)
{
    float epsilon = 1e-6;
    float scale = 1.0f / epsilon;

    int w = grayRef.cols;
    int h = grayRef.rows;
    int n = w*h;

    // calculate per-pixel residuals
    calculateError(grayRef, depthRef, grayCur, depthCur, xi, K, residuals);

    // create and fill Jacobian column by column
    float* residualsInc = new float[n];
    for (int j = 0; j < 6; ++j)
    {
        Eigen::VectorXf unitVec = Eigen::VectorXf::Zero(6);
        unitVec[j] = epsilon;

        // left-multiplicative increment on SE3
        Eigen::VectorXf xiEps = Sophus::SE3f::log(Sophus::SE3f::exp(unitVec) * Sophus::SE3f::exp(xi));

        calculateError(grayRef, depthRef, grayCur, depthCur, xiEps, K, residualsInc);
        for (int i = 0; i < n; ++i)
            J[i*6 + j] = (residualsInc[i] - residuals[i]) * scale;
    }
    delete[] residualsInc;
}


void DVO::compute_JtR(const float* J, const float* residuals, Vec6f &b, int validRows)
{
    int n = 6;
    int m = validRows;

    // compute b = Jt*r
    for (int j = 0; j < n; ++j)
    {
        float val = 0.0f;
        for (int i = 0; i < m; ++i)
            val += J[i*6 + j] * residuals[i];
        b[j] = val;
    }
}


void DVO::compute_JtJ(const float* J, Mat6f &A, const float* weights, int validRows, bool useWeights)
{
    int n = 6;
    int m = validRows;

    // compute A = Jt*J
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            float val = 0.0f;
            for (int i = 0; i < m; ++i)
            {
                float valSqr = J[i*6 + j] * J[i*6 + k];
                if (useWeights)
                    valSqr *= weights[i];
                val += valSqr;
            }
            A(k, j) = val;
        }
    }
}


void DVO::deriveAnalytic(const cv::Mat &grayRef, const cv::Mat &depthRef,
                   const cv::Mat &grayCur, const cv::Mat &depthCur,
                   const cv::Mat &gradX, const cv::Mat &gradY,
                   const Eigen::VectorXf &xi, const Eigen::Matrix3f &K,
                   float* residuals, float* J)
{
    // reference input images
    int w = grayRef.cols;
    int h = grayRef.rows;
    int n = w*h;
    const float* ptrDepthRef = (const float*)depthRef.data;

    // camera intrinsics
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;

    // convert SE3 to rotation matrix and translation vector
    Eigen::Matrix3f rotMat;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rotMat, t);

    // calculate per-pixel residuals
    calculateError(grayRef, depthRef, grayCur, depthCur, xi, K, residuals);

    // reference gradient images
    const float* ptrGradX = (const float*)gradX.data;
    const float* ptrGradY = (const float*)gradY.data;

    // create and fill Jacobian row by row
    float residualRowJ[6];
    for (size_t y = 0; y < h; ++y)
    {
        for (size_t x = 0; x < w; ++x)
        {
            size_t off = y*w + x;

            // project 2d point back into 3d using its depth
            float dRef = ptrDepthRef[y*w + x];
            if (dRef > 0.0f)
            {
                float x0 = (static_cast<float>(x) - cx) * fxInv;
                float y0 = (static_cast<float>(y) - cy) * fyInv;
                float scale = 1.0f;
                //scale = std::sqrt(x0*x0 + y0*y0 + 1.0);
                dRef = dRef * scale;
                x0 = x0 * dRef;
                y0 = y0 * dRef;

                // transform reference 3d point into current frame
                // reference 3d point
                Eigen::Vector3f pt3Ref(x0, y0, dRef);
                Eigen::Vector3f pt3 = rotMat * pt3Ref + t;
                if (pt3[2] > 0.0f)
                {
                    // project 3d point to 2d
                    Eigen::Vector3f pt2CurH = K * pt3;
                    float ptZinv = 1.0f / pt2CurH[2];
                    float px = pt2CurH[0] * ptZinv;
                    float py = pt2CurH[1] * ptZinv;

                    // compute interpolated image gradient
                    float dX = interpolate(ptrGradX, px, py, w, h);
                    float dY = interpolate(ptrGradY, px, py, w, h);
                    if (!std::isnan(dX) && !std::isnan(dY))
                    {
                        dX = fx * dX;
                        dY = fy * dY;
                        float pt3Zinv = 1.0f / pt3[2];

                        // shorter computation
                        residualRowJ[0] = dX * pt3Zinv;
                        residualRowJ[1] = dY * pt3Zinv;
                        residualRowJ[2] = - (dX * pt3[0] + dY * pt3[1]) * pt3Zinv * pt3Zinv;
                        residualRowJ[3] = - (dX * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv - dY * (1 + (pt3[1] * pt3Zinv) * (pt3[1] * pt3Zinv));
                        residualRowJ[4] = + dX * (1.0 + (pt3[0] * pt3Zinv) * (pt3[0] * pt3Zinv)) + (dY * pt3[0] * pt3[1]) * pt3Zinv * pt3Zinv;
                        residualRowJ[5] = (- dX * pt3[1] + dY * pt3[0]) * pt3Zinv;
                    }
                }
            }

            // set 1x6 Jacobian row for current residual
            // invert Jacobian according to kerl2012msc.pdf (necessary?)
            for (int j = 0; j < 6; ++j)
                J[off*6 + j] = - residualRowJ[j];
        }
    }
}


void DVO::buildPyramid(const cv::Mat &depth, const cv::Mat &gray, std::vector<cv::Mat> &depthPyramid, std::vector<cv::Mat> &grayPyramid)
{
    grayPyramid.push_back(gray);
    depthPyramid.push_back(depth);

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        // downsample grayscale image
        cv::Mat grayDown = downsampleGray(grayPyramid[i-1]);
        grayPyramid.push_back(grayDown);

        // downsample depth image
        cv::Mat depthDown = downsampleDepth(depthPyramid[i-1]);
        depthPyramid.push_back(depthDown);
    }
}


void DVO::align(const cv::Mat &depthRef, const cv::Mat &grayRef, const cv::Mat &depthCur, const cv::Mat &grayCur, Eigen::Matrix4f &pose)
{
    // downsampling
    std::vector<cv::Mat> grayRefPyramid;
    std::vector<cv::Mat> depthRefPyramid;
    buildPyramid(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    std::vector<cv::Mat> grayCurPyramid;
    std::vector<cv::Mat> depthCurPyramid;
    buildPyramid(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

    align(depthRefPyramid, grayRefPyramid, depthCurPyramid, grayCurPyramid, pose);
}


void DVO::align(const std::vector<cv::Mat> &depthRefPyramid, const std::vector<cv::Mat> &grayRefPyramid,
                const std::vector<cv::Mat> &depthCurPyramid, const std::vector<cv::Mat> &grayCurPyramid,
                Eigen::Matrix4f &pose)
{
    Vec6f xi;
    convertTfToSE3(pose, xi);

    Vec6f lastXi = Vec6f::Zero();

    int maxLevel = numPyramidLevels_-1;
    int minLevel = 1;
    float initGradDescStepSize = 1e-3f;
    float gradDescStepSize = initGradDescStepSize;

    Mat6f A;
    Mat6f diagMatA = Mat6f::Identity();
    Vec6f delta;

    for (int lvl = maxLevel; lvl >= minLevel; --lvl)
    {
        float lambda = 0.1f;

        int w = sizePyramid_[lvl].width;
        int h = sizePyramid_[lvl].height;
        int n = w*h;

        cv::Mat grayRef = grayRefPyramid[lvl];
        cv::Mat depthRef = depthRefPyramid[lvl];
        cv::Mat grayCur = grayCurPyramid[lvl];
        cv::Mat depthCur = depthCurPyramid[lvl];
        Eigen::Matrix3f kLevel = kPyramid_[lvl];
        //std::cout << "level " << level << " (size " << depthRef.cols << "x" << depthRef.rows << ")" << std::endl;

        // compute gradient images
        computeGradient(grayCur, gradX_[lvl], 0);
        computeGradient(grayCur, gradY_[lvl], 1);

        float errorLast = std::numeric_limits<float>::max();
        for (int itr = 0; itr < numIterations_; ++itr)
        {
            // compute residuals and Jacobian
#if 0
            deriveNumeric(grayRef, depthRef, grayCur, depthCur, xi, kLevel, residuals_[lvl], J_[lvl]);
#else
            deriveAnalytic(grayRef, depthRef, grayCur, depthCur, gradX_[lvl], gradY_[lvl], xi, kLevel, residuals_[lvl], J_[lvl]);
#endif

#if 0
            // compute and show error image
            cv::Mat errorImage;
            calculateErrorImage(residuals_[level], grayRef.cols, grayRef.rows, errorImage);
            std::stringstream ss;
            ss << dataFolder << "residuals_" << level << "_";
            ss << std::setw(2) << std::setfill('0') << itr << ".png";
            cv::imwrite(ss.str(), errorImage);
            cv::imshow("error", errorImage);
            cv::waitKey(100);
#endif

            // calculate error
            float error = calculateError(residuals_[lvl], n);

            if (useWeights_)
            {
                // compute robust weights
                computeWeights(residuals_[lvl], weights_[lvl], n);
                // apply robust weights
                applyWeights(weights_[lvl], residuals_[lvl], n);
            }

            // compute update
            Vec6f b;
            compute_JtR(J_[lvl], residuals_[lvl], b, n);

            if (algo_ == GradientDescent)
            {
                // Gradient Descent
                delta = -gradDescStepSize * b * (1.0f / b.norm());
            }
            else if (algo_ == GaussNewton)
            {
                // Gauss-Newton algorithm
                compute_JtJ(J_[lvl], A, weights_[lvl], n, useWeights_);
                // solve using Cholesky LDLT decomposition
                delta = -(A.ldlt().solve(b));
            }
            else if (algo_ == LevenbergMarquardt)
            {
                // Levenberg-Marquardt algorithm
                compute_JtJ(J_[lvl], A, weights_[lvl], n, useWeights_);
                diagMatA.diagonal() = lambda * A.diagonal();
                delta = -((A + diagMatA).ldlt().solve(b));
            }

            // apply update: left-multiplicative increment on SE3
            lastXi = xi;
            xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta) * Sophus::SE3f::exp(xi));
#if 0
            std::cout << "delta = " << delta.transpose() << " size = " << delta.rows() << " x " << delta.cols() << std::endl;
            std::cout << "xi = " << xi.transpose() << std::endl;
#endif

            // compute error again
            error = calculateError(residuals_[lvl], n);

            if (algo_ == LevenbergMarquardt)
            {
                if (error >= errorLast)
                {
                    lambda = lambda * 5.0f;
                    xi = lastXi;

                    if (lambda > 5.0f)
                        break;
                }
                else
                {
                    lambda = lambda / 1.5f;
                }
            }
            else if (algo_ == GaussNewton)
            {
                // break if no improvement (0.99 or 0.995)
                if (error / errorLast > 0.995f)
                    break;
            }
            else if (algo_ == GradientDescent)
            {
                if (error >= errorLast)
                {
                    gradDescStepSize = gradDescStepSize * 0.5f;
                    if (gradDescStepSize <= initGradDescStepSize * 0.01f)
                        gradDescStepSize = initGradDescStepSize * 0.01f;
                    xi = lastXi;
                }
                else
                {
                    gradDescStepSize = gradDescStepSize * 2.0f;
                    if (gradDescStepSize >= initGradDescStepSize * 100.0f)
                        gradDescStepSize = initGradDescStepSize * 100.0f;

                    // break if no improvement (0.99 or 0.995)
                    if (error / errorLast > 0.995f)
                        break;
                }
            }

            errorLast = error;
        }
    }

    // store to output pose
    convertSE3ToTf(xi, pose);
}
