#include "semantic_slam/SemanticKeyframe.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/ImuBiasNode.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <unordered_map>

SemanticKeyframe::SemanticKeyframe(Key key,
                                   ros::Time time,
                                   bool include_inertial)
  : key_(key)
  , time_(time)
  , in_graph_(false)
  , measurements_processed_(false)
  , covariance_computed_exactly_(false)
  , loop_closing_(false)
  , include_inertial_(include_inertial)
{
    graph_node_ = util::allocate_aligned<SE3Node>(key, time);
    pose_covariance_ = Eigen::MatrixXd::Zero(6, 6);

    // Should just be passing in the index not the key, need to hack the index
    // out here
    auto index = Symbol(key).index();

    if (include_inertial_) {
        velocity_node_ =
          util::allocate_aligned<VectorNode<3>>(Symbol('v', index), time);
        bias_node_ =
          util::allocate_aligned<ImuBiasNode>(Symbol('b', index), time);
    }
}

void
SemanticKeyframe::addToGraph(boost::shared_ptr<FactorGraph> graph)
{
    graph->addNode(graph_node_);
    if (include_inertial_) {
        graph->addNode(velocity_node_);
        graph->addNode(bias_node_);
    }

    graph->addFactor(spine_factor_);

    if (graph->solver_options().linear_solver_ordering) {
        graph_node_->addToOrderingGroup(
          graph->solver_options().linear_solver_ordering, 10);
    }

    in_graph_ = true;
}

void
SemanticKeyframe::addConnection(SemanticKeyframe::Ptr other, int count)
{
    neighbors_[other] = count;
}

void
SemanticKeyframe::addGeometricConnection(SemanticKeyframe::Ptr other, int count)
{
    geometric_neighbors_[other] = count;
}

void
SemanticKeyframe::updateConnections()
{
    // TODO if we previously had a connection to a neighbor that we removed in
    // the graph, this doesn't properly account for that (it won't update with
    // neighbor's connections list with the now-removed link information)

    neighbors_.clear();

    for (const EstimatedObject::Ptr& obj : visible_objects_) {
        for (const SemanticKeyframe::Ptr& kf : obj->keyframe_observations()) {
            neighbors_[kf]++;
        }
    }

    for (auto neighbor_weight : neighbors_) {
        neighbor_weight.first->addConnection(shared_from_this(),
                                             neighbor_weight.second);
    }
}

void
SemanticKeyframe::updateGeometricConnections()
{
    geometric_neighbors_.clear();

    for (const GeometricFeature::Ptr& feat : visible_geometric_features_) {
        if (!feat->active)
            continue;
        for (const SemanticKeyframe::Ptr& kf : feat->keyframe_observations) {
            geometric_neighbors_[kf]++;
        }
    }

    for (auto neighbor_weight : geometric_neighbors_) {
        neighbor_weight.first->addGeometricConnection(shared_from_this(),
                                                      neighbor_weight.second);
    }
}