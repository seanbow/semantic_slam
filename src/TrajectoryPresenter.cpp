#include "semantic_slam/TrajectoryPresenter.h"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Quaternion.h>
#include <visualization_msgs/Marker.h>

void TrajectoryPresenter::setup()
{
    publisher_ = nh_.advertise<visualization_msgs::Marker>("trajectory", 10);
}

void TrajectoryPresenter::present()
{
    // Find the last odometry node N and assume that all in [0,N] exist
    NodeInfoPtr node = graph_->findLastNode('x');
    size_t last_index = node->index();
    
    
    visualization_msgs::Marker traj;
    traj.type = visualization_msgs::Marker::LINE_STRIP;

    // initialize quaternion to identity. this isn't actually used anywhere but it suppresses a warning
    traj.pose.orientation.x = 0;
    traj.pose.orientation.y = 0;
    traj.pose.orientation.z = 0;
    traj.pose.orientation.w = 1;

    traj.header.frame_id = "/map";
    traj.header.stamp = *node->time();

    Eigen::Vector3d rgb_color(0, 1, 0);
    double Z_SCALE = 1.0;

    double scale = 0.25;
    traj.scale.x = scale;
    traj.scale.y = scale;
    traj.scale.z = scale;
    traj.color.r = rgb_color(0);
    traj.color.g = rgb_color(1);
    traj.color.b = rgb_color(2);
    traj.color.a = 1.0f;

    traj.ns = "optimized_trajectory";
    traj.id = 0;
    traj.action = visualization_msgs::Marker::ADD;

    // create vertices for each pose
    for (size_t i = 0; i < last_index; ++i) {
        gtsam::Pose3 pose;

        bool got_estimate = graph_->getEstimate(gtsam::Symbol('x', i), pose);

        if (!got_estimate) continue;

        gtsam::Point3 p = pose.translation();

        geometry_msgs::Point p_msg;
        p_msg.x = p.x();
        p_msg.y = p.y();
        p_msg.z = Z_SCALE * p.z();

        traj.points.push_back(p_msg);
    }

    publisher_.publish(traj);
}
