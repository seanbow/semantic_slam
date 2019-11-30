#include "semantic_slam/GeometricMapPresenter.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/FactorGraph.h"

#include <visualization_msgs/Marker.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Quaternion.h>

void GeometricMapPresenter::setup()
{
    pub_ = nh_.advertise<visualization_msgs::Marker>("map_points", 10);
}

void GeometricMapPresenter::present(const std::vector<SemanticKeyframe::Ptr>& keyframes,
                                  const std::vector<EstimatedObject::Ptr>& objects)
{
    visualization_msgs::Marker points;
    points.type = visualization_msgs::Marker::POINTS;

    // initialize quaternion to identity. this isn't actually used anywhere but it suppresses a warning
    points.pose.orientation.x = 0;
    points.pose.orientation.y = 0;
    points.pose.orientation.z = 0;
    points.pose.orientation.w = 1;

    points.header.frame_id = "/map";
    points.header.stamp = ros::Time::now();

    Eigen::Vector3d rgb_color(153, 0, 76);
    rgb_color /= 255;

    double scale = 1;
    points.scale.x = scale;
    points.scale.y = scale;
    points.color.r = rgb_color(0);
    points.color.g = rgb_color(1);
    points.color.b = rgb_color(2);
    points.color.a = 1.0f;

    points.ns = "geometric_features";
    points.id = 0;
    points.action = visualization_msgs::Marker::ADD;

    auto last_node = graph_->findLastNode('g');

    if (!last_node) return;

    size_t last_index = last_node->index();

    for (size_t i = 0; i < last_index; ++i) {

        auto node = graph_->getNode<Vector3dNode>(Symbol('g', i));

        if (!node || !!node->active() || !node->vector().allFinite()) continue;

        if (node->vector().norm() > 1e4) continue;

        geometry_msgs::Point p_msg;
        p_msg.x = node->vector()(0);
        p_msg.y = node->vector()(1);
        p_msg.z = node->vector()(2);

        points.points.push_back(p_msg);
    }

    pub_.publish(points);
}
