#pragma once

#include <ceres/ceres.h>
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/geometry.h"

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
                         double lambda=1.0,
                         int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

private:
    geometry::ObjectModelBasis model_;

    SE3NodePtr object_node_;
    std::vector<Vector3dNodePtr> landmark_nodes_;
    VectorXdNodePtr coefficient_node_;

    double lambda_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresStructureFactorPtr = CeresStructureFactor::Ptr;
