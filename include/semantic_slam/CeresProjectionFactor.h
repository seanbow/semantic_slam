#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"

namespace gtsam {
class NonlinearFactor;
}

class CameraCalibration;

class CeresProjectionFactor : public CeresFactor
{
  public:
    using This = CeresProjectionFactor;
    using Ptr = boost::shared_ptr<This>;

    CeresProjectionFactor(boost::shared_ptr<SE3Node> camera_node,
                          boost::shared_ptr<Vector3dNode> landmark_node,
                          const Eigen::Vector2d& image_coords,
                          const Eigen::Matrix2d& msmt_covariance,
                          boost::shared_ptr<CameraCalibration> calibration,
                          const Pose3& body_T_sensor,
                          bool use_huber = true,
                          int tag = 0);

    ~CeresProjectionFactor();

    boost::shared_ptr<SE3Node> camera_node() const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[0]);
    }
    boost::shared_ptr<Vector3dNode> landmark_node() const
    {
        return boost::static_pointer_cast<Vector3dNode>(nodes_[1]);
    }

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    CeresFactor::Ptr clone() const;

    void createGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    Eigen::Vector2d image_coords_;
    Eigen::Matrix2d covariance_;
    boost::shared_ptr<CameraCalibration> calibration_;
    Pose3 body_T_sensor_;

    bool robust_loss_;

    mutable boost::shared_ptr<gtsam::NonlinearFactor> gtsam_factor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresProjectionFactorPtr = CeresProjectionFactor::Ptr;
