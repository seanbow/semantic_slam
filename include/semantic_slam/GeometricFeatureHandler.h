#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/feature_tracker/FeatureTracker.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/CameraCalibration.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <thread>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <condition_variable>
#include <sensor_msgs/Image.h>

class SmartProjectionFactor;
class MultiProjectionFactor;

class GeometricFeatureHandler : public Handler 
{
public:
    void setup();
    void update();

    void addKeyframe(const SemanticKeyframe::Ptr& keyframe);

    void loadParameters();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    void setExtrinsicCalibration(const Pose3& body_T_sensor);

    void processPendingFrames();

    std::unordered_map<int, boost::shared_ptr<Vector3dNode>> landmark_nodes() { return landmark_nodes_; }

    std::unordered_map<int, boost::shared_ptr<MultiProjectionFactor>> factors() { return multi_factors_; }

private:

    image_transport::Subscriber img_sub_;

    Pose3 I_T_C_;
    boost::shared_ptr<CameraCalibration> calibration_;

    int last_img_seq_;
    std::deque<sensor_msgs::ImageConstPtr> img_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    double reprojection_error_threshold_;
    double cam_sigma_;

    bool use_smart_projection_factors_;

    // std::deque<ros::Time> keyframe_time_queue_;
    // std::mutex keyframe_mutex_;
    // std::condition_variable keyframe_cv_;

    boost::shared_ptr<FeatureTracker> tracker_;
    FeatureTracker::Params tracker_params_;

    std::unordered_map<int, boost::shared_ptr<Vector3dNode>> landmark_nodes_;

    std::unordered_map<int, boost::shared_ptr<SmartProjectionFactor>> smart_factors_;
    std::unordered_map<int, boost::shared_ptr<MultiProjectionFactor>> multi_factors_;

    std::deque<SemanticKeyframe::Ptr> kfs_to_process_;

    bool running_;
    std::thread work_thread_;

    void extractKeypointsThread();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};