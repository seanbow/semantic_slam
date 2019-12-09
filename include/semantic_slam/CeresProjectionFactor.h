#pragma once

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/pose_math.h"

namespace gtsam {
class NonlinearFactor;
}

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
                          bool use_huber = true,
                          int tag = 0);

    ~CeresProjectionFactor();

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    boost::shared_ptr<gtsam::NonlinearFactor> getGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    SE3NodePtr camera_node_;
    Vector3dNodePtr landmark_node_;

    Eigen::Vector2d image_coords_;
    Eigen::Matrix2d covariance_;
    boost::shared_ptr<CameraCalibration> calibration_;
    Pose3 body_T_sensor_;

    bool robust_loss_;

    boost::shared_ptr<gtsam::NonlinearFactor> gtsam_factor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresProjectionFactorPtr = CeresProjectionFactor::Ptr;
