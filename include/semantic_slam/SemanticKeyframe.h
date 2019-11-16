#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <ros/ros.h>
#include <vector>
#include <map>
#include <eigen3/Eigen/Core>

#include <boost/enable_shared_from_this.hpp>

class SemanticKeyframe : public boost::enable_shared_from_this<SemanticKeyframe> {
public:

    using This = SemanticKeyframe;
    using Ptr = boost::shared_ptr<This>;

    SemanticKeyframe(Key key, ros::Time time);

    Key key() const { return key_; }
    int index() const { return Symbol(key_).index(); }
    unsigned char chr() const { return Symbol(key_).chr(); }

    ros::Time time() const { return time_; }

    Pose3& pose() { return pose_; }
    const Pose3& pose() const { return pose_; }

    Eigen::MatrixXd& covariance() { return pose_covariance_; }
    const Eigen::MatrixXd& covariance() const { return pose_covariance_; }

    Pose3& odometry() { return odometry_; }
    const Pose3& odometry() const { return odometry_; }

    CeresFactorPtr& spine_factor() { return spine_factor_; }
    const CeresFactorPtr& spine_factor() const { return spine_factor_; }

    SE3NodePtr& graph_node() { return graph_node_; }
    const SE3NodePtr& graph_node() const { return graph_node_; }

    bool inGraph() const { return in_graph_; }

    void addToGraph(boost::shared_ptr<FactorGraph> graph);

    void updateConnections();

    void addConnection(SemanticKeyframe::Ptr other, int weight);

    aligned_vector<ObjectMeasurement> measurements;

    const std::map<SemanticKeyframe::Ptr, int> neighbors() const { return neighbors_; }

    // std::vector<ObjectMeasurement>& measurements() { return measurements_; }
    std::vector<EstimatedObject::Ptr>& visible_objects() { return visible_objects_; }

private:
    Key key_;
    ros::Time time_;

    bool in_graph_;

    Pose3 odometry_;
    Pose3 pose_;
    Eigen::MatrixXd pose_covariance_;

    SE3NodePtr graph_node_;

    CeresFactorPtr spine_factor_;

    // aligned_vector<ObjectMeasurement> measurements_;
    std::vector<EstimatedObject::Ptr> visible_objects_;

    // Connections to keyframes that observe the same objects along with the number of
    // mutually observed objects
    // can't use unordered_map without custom hash function bc std::hash<boost::shared_ptr> is not defined
    std::map<SemanticKeyframe::Ptr, int> neighbors_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
