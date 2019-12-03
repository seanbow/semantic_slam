#include "semantic_slam/CeresStructureFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_structure.h"

#include "semantic_slam/keypoints/gtsam/StructureFactor.h"

CeresStructureFactor::CeresStructureFactor(SE3NodePtr object_node,
                         std::vector<Vector3dNodePtr> landmark_nodes,
                         VectorXdNodePtr coefficient_node,
                         const geometry::ObjectModelBasis& model,
                         const Eigen::VectorXd& weights,
                         double lambda,
                         int tag)
    : CeresFactor(FactorType::STRUCTURE, tag),
      model_(model),
      object_node_(object_node),
      landmark_nodes_(landmark_nodes),
      coefficient_node_(coefficient_node)
{
    cf_ = StructureCostTerm::Create(model, weights, lambda);

    // gtsam support
    std::vector<Key> landmark_keys;
    for (auto l : landmark_nodes) {
        landmark_keys.push_back(l->key());
    }

    Key coefficient_key = 0;
    if (coefficient_node) {
        coefficient_key = coefficient_node->key();
    }

    gtsam_factor_ = util::allocate_aligned<semslam::StructureFactor>(
        object_node->key(),
        landmark_keys,
        coefficient_key,
        model,
        weights,
        lambda
    );
}

void CeresStructureFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // Accumulate the parameter blocks...
    std::vector<double*> blocks;
    blocks.push_back(object_node_->pose().rotation_data());
    blocks.push_back(object_node_->pose().translation_data());

    for (size_t i = 0; i < landmark_nodes_.size(); ++i) {
        blocks.push_back(landmark_nodes_[i]->vector().data());
    }

    if (coefficient_node_) {
        blocks.push_back(coefficient_node_->vector().data());
    }

    residual_id_ = problem->AddResidualBlock(cf_, NULL, blocks);
}

void CeresStructureFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->RemoveResidualBlock(residual_id_);
}


boost::shared_ptr<gtsam::NonlinearFactor> 
CeresStructureFactor::getGtsamFactor() const
{
    return gtsam_factor_;
}