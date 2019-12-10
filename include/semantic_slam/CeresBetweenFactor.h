#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/pose_math.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>

namespace gtsam {
class NonlinearFactor;
}

class CeresBetweenFactor : public CeresFactor
{
  public:
    CeresBetweenFactor(SE3NodePtr node0,
                       SE3NodePtr node1,
                       Pose3 between,
                       Eigen::MatrixXd covariance,
                       int tag = 0);
    ~CeresBetweenFactor();

    SE3NodePtr node0() const { return boost::static_pointer_cast<SE3Node>(nodes_[0]); }
    SE3NodePtr node1() const { return boost::static_pointer_cast<SE3Node>(nodes_[1]); }

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    boost::shared_ptr<gtsam::NonlinearFactor> getGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

    CeresFactor::Ptr clone() const;

    using This = CeresBetweenFactor;
    using Ptr = boost::shared_ptr<This>;

  private:
    Pose3 between_;
    Eigen::MatrixXd covariance_;

    boost::shared_ptr<gtsam::NonlinearFactor> gtsam_factor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresBetweenFactorPtr = CeresBetweenFactor::Ptr;