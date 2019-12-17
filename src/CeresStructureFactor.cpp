#include "semantic_slam/CeresStructureFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_structure.h"

#include "semantic_slam/keypoints/gtsam/StructureFactor.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

CeresStructureFactor::CeresStructureFactor(
  SE3NodePtr object_node,
  std::vector<Vector3dNodePtr> landmark_nodes,
  VectorXdNodePtr coefficient_node,
  const geometry::ObjectModelBasis& model,
  const Eigen::VectorXd& weights,
  double lambda,
  int tag)
  : CeresFactor(FactorType::STRUCTURE, tag)
  , model_(model)
  , weights_(weights)
  , lambda_(lambda)
{
    cf_ = StructureCostTerm::Create(model, weights, lambda);

    m_ = model.mu.cols();
    k_ = model.pc.rows() / 3;

    nodes_.push_back(object_node);
    for (auto& n : landmark_nodes)
        nodes_.push_back(n);
    if (coefficient_node)
        nodes_.push_back(coefficient_node);

    createGtsamFactor();
}

void
CeresStructureFactor::createGtsamFactor() const
{
    if (object_node() && landmark_node(0)) {
        std::vector<Key> landmark_keys;
        for (int i = 0; i < m_; ++i) {
            landmark_keys.push_back(landmark_node(i)->key());
        }

        Key coefficient_key = 0;
        if (k_ > 0) {
            coefficient_key = coefficient_node()->key();
        }

        gtsam_factor_ =
          util::allocate_aligned<semslam::StructureFactor>(object_node()->key(),
                                                           landmark_keys,
                                                           coefficient_key,
                                                           model_,
                                                           weights_,
                                                           lambda_);
    }
}

CeresFactor::Ptr
CeresStructureFactor::clone() const
{
    // ugh
    std::vector<Vector3dNodePtr> new_landmark_placeholder(model_.mu.cols(),
                                                          nullptr);

    return util::allocate_aligned<CeresStructureFactor>(
      nullptr, new_landmark_placeholder, nullptr, model_, weights_, lambda_);
}

void
CeresStructureFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // Accumulate the parameter blocks...
    std::vector<double*> blocks;
    blocks.push_back(object_node()->pose().data());

    for (size_t i = 0; i < m_; ++i) {
        blocks.push_back(landmark_node(i)->vector().data());
    }

    if (k_ > 0) {
        blocks.push_back(coefficient_node()->vector().data());
    }

    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_, NULL, blocks);
    residual_ids_.[problem.get()] = residual_id;

    active_ = true;
}

void
CeresStructureFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    if (!gtsam_factor_)
        createGtsamFactor();

    graph->push_back(gtsam_factor_);
}