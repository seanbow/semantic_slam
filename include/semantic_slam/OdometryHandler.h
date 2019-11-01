#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"

#include <nav_msgs/Odometry.h>
#include <mutex>
#include <deque>
#include <gtsam/geometry/Pose3.h>

class OdometryHandler : public Handler
{
public:
    void setup();

    void update();

    void msgCallback(const nav_msgs::Odometry::ConstPtr& msg);

    gtsam::Pose3 msgToPose3(const nav_msgs::Odometry& msg);

    // inherit constructor
    using Handler::Handler;

private:
    ros::Subscriber subscriber_;

	std::deque<nav_msgs::Odometry> msg_queue_;

    std::mutex mutex_;

    gtsam::Pose3 last_odom_;
    ros::Time last_time_;
    
    size_t received_msgs_;
    size_t last_msg_seq_;

    unsigned char node_chr_;
    ros::Duration node_period_;

};
