#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/geometry.h"
#include <ceres/ceres.h>

namespace semslam {
class StructureFactor;
}

class CeresStructureFactor : public CeresFactor
{
  public:
    using This = CeresStructureFactor;
    using Ptr = boost::shared_ptr<This>;

    CeresStructureFactor(SE3NodePtr object_node,
                         std::vector<Vector3dNodePtr> landmark_nodes,
                         VectorXdNodePtr coefficient_node,
                         const geometry::ObjectModelBasis& model,
                         const Eigen::VectorXd& weights,
                         double lambda = 1.0,
                         int tag = 0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    SE3NodePtr object_node() const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[0]);
    }

    Vector3dNodePtr landmark_node(int i) const
    {
        return boost::static_pointer_cast<Vector3dNode>(nodes_[i + 1]);
    }

    VectorXdNodePtr coefficient_node() const
    {
        return boost::static_pointer_cast<VectorXdNode>(nodes_[m_ + 1]);
    }

    CeresFactor::Ptr clone() const;

    void createGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    geometry::ObjectModelBasis model_;

    size_t m_, k_;

    Eigen::VectorXd weights_;
    double lambda_;

    mutable boost::shared_ptr<semslam::StructureFactor> gtsam_factor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresStructureFactorPtr = CeresStructureFactor::Ptr;
