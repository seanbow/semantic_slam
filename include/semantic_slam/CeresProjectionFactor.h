#pragma once

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/CameraCalibration.h"

class CeresProjectionFactor : public CeresFactor
{
public:
    using This = CeresProjectionFactor;
    using Ptr = boost::shared_ptr<This>;

    CeresProjectionFactor(SE3NodePtr camera_node,
                          Vector3dNodePtr landmark_node,
                          const Eigen::Vector2d& image_coords,
                          const Eigen::Matrix2d& msmt_covariance,
                          boost::shared_ptr<CameraCalibration> calibration,
                          const Pose3& body_T_sensor,
                          int tag = 0);

    ~CeresProjectionFactor();

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

private:
    SE3NodePtr camera_node_;
    Vector3dNodePtr landmark_node_;

};

using CeresProjectionFactorPtr = CeresProjectionFactor::Ptr;
