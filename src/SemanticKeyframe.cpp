#include "semantic_slam/SemanticKeyframe.h"

#include <unordered_map>

SemanticKeyframe::SemanticKeyframe(Key key, ros::Time time)
    : key_(key),
      time_(time),
      in_graph_(false),
      measurements_processed_(false),
      covariance_computed_exactly_(false)
{
    graph_node_ = util::allocate_aligned<SE3Node>(key, time);
    pose_covariance_ = Eigen::MatrixXd::Zero(6,6);
}

void SemanticKeyframe::addToGraph(boost::shared_ptr<FactorGraph> graph)
{
    graph->addNode(graph_node_);
    graph->addFactor(spine_factor_);
    in_graph_ = true;
}

void SemanticKeyframe::addConnection(SemanticKeyframe::Ptr other, int count)
{
    neighbors_[other] = count;
}

void SemanticKeyframe::addGeometricConnection(SemanticKeyframe::Ptr other, int count)
{
    geometric_neighbors_[other] = count;
}

void SemanticKeyframe::updateConnections()
{
    neighbors_.clear();

    for (const EstimatedObject::Ptr& obj : visible_objects_) {
        for (const SemanticKeyframe::Ptr& kf : obj->keyframe_observations()) {
            neighbors_[kf]++;
        }
    }

    for (auto neighbor_weight : neighbors_) {
        neighbor_weight.first->addConnection(shared_from_this(), neighbor_weight.second);
    }
}

void SemanticKeyframe::updateGeometricConnections()
{
    geometric_neighbors_.clear();

    for (const GeometricFeature::Ptr& feat : visible_geometric_features_) {
        if (!feat->active) continue;
        for (const SemanticKeyframe::Ptr& kf : feat->keyframe_observations) {
            geometric_neighbors_[kf]++;
        }
    }

    for (auto neighbor_weight : geometric_neighbors_) {
        neighbor_weight.first->addGeometricConnection(shared_from_this(), neighbor_weight.second);
    }
}