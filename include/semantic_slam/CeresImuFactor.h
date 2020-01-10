#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"

namespace gtsam {
class NonlinearFactor;
}

class InertialIntegrator;

class CeresImuFactor : public CeresFactor
{
  public:
    using This = CeresImuFactor;
    using Ptr = boost::shared_ptr<This>;

    CeresImuFactor(boost::shared_ptr<SE3Node> pose0,
                   boost::shared_ptr<Vector3Node> vel0,
                   boost::shared_ptr<VectorNode<6>> bias0,
                   boost::shared_ptr<SE3Node> pose1,
                   boost::shared_ptr<Vector3Node> vel1,
                   boost::shared_ptr<VectorNode<6>> bias1,
                   boost::shared_ptr<InertialIntegrator> integrator,
                   int tag = 0);

    ~CeresImuFactor();

    boost::shared_ptr<SE3Node> pose_node(int i) const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[3 * i]);
    }

    boost::shared_ptr<Vector3Node> vel_node(int i) const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[3 * i + 1]);
    }

    boost::shared_ptr<VectorNode<6>> bias_node(int i) const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[3 * i + 2]);
    }

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    CeresFactor::Ptr clone() const;

    void createGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    mutable boost::shared_ptr<gtsam::NonlinearFactor> gtsam_factor_;

    boost::shared_ptr<InertialIntegrator> integrator_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresImuFactorPtr = CeresImuFactor::Ptr;
