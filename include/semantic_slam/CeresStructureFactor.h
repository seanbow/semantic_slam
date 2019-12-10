#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/pose_math.h"
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

    CeresFactor::Ptr clone() const;

    boost::shared_ptr<gtsam::NonlinearFactor> getGtsamFactor() const;
    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    geometry::ObjectModelBasis model_;

    SE3NodePtr object_node_;
    std::vector<Vector3dNodePtr> landmark_nodes_;
    VectorXdNodePtr coefficient_node_;

    Eigen::VectorXd weights_;
    double lambda_;

    boost::shared_ptr<semslam::StructureFactor> gtsam_factor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresStructureFactorPtr = CeresStructureFactor::Ptr;
