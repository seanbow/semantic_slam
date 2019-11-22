#pragma once

#include "semantic_slam/Handler.h"
#include "semantic_slam/feature_tracker/FeatureTracker.h"

#include <ros/ros.h>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <sensor_msgs/Image.h>

class GeometricFeatureHandler : public Handler 
{
public:
    void setup();
    void update();

    void addKeyframeTime(ros::Time t);

    void loadParameters();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:

    ros::Subscriber img_sub_;

    int last_img_seq_;
    std::deque<sensor_msgs::ImageConstPtr> img_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    boost::shared_ptr<FeatureTracker> tracker_;
    FeatureTracker::Params tracker_params_;

    bool running_;
    std::thread work_thread_;

    void extractKeypointsThread();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};