#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/Pose3.h"

class SE3Node;

class CeresSE3PriorFactor : public CeresFactor
{
  public:
    CeresSE3PriorFactor(boost::shared_ptr<SE3Node> node,
                        const Pose3& prior,
                        const Eigen::MatrixXd& covariance,
                        int tag = 0);
    ~CeresSE3PriorFactor();

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    boost::shared_ptr<SE3Node> node() const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[0]);
    }

    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

    // Returns a new factor that is identical to this one except in which node
    // it operates on
    CeresFactor::Ptr clone() const;

    using Ptr = boost::shared_ptr<CeresSE3PriorFactor>;

  private:
    Pose3 prior_;
    Eigen::MatrixXd covariance_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresSE3PriorFactorPtr = CeresSE3PriorFactor::Ptr;
