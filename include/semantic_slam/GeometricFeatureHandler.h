#pragma once

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/feature_tracker/FeatureTracker.h"

#include <condition_variable>
#include <deque>
#include <image_transport/image_transport.h>
#include <mutex>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <thread>
#include <unordered_map>

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

    void removeLandmark(int index);

    void updateEssentialGraph(
      const std::vector<SemanticKeyframe::Ptr>& keyframes_processed);

    std::unordered_map<int, boost::shared_ptr<Vector3dNode>> landmark_nodes()
    {
        return landmark_nodes_;
    }

    std::unordered_map<int, boost::shared_ptr<CeresFactor>> factors()
    {
        return factors_;
    }

  private:
    image_transport::Subscriber img_sub_;

    Pose3 I_T_C_;
    boost::shared_ptr<CameraCalibration> calibration_;

    size_t last_img_seq_;
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

    std::unordered_map<int, boost::shared_ptr<GeometricFeature>> features_;
    std::unordered_map<int, boost::shared_ptr<Vector3dNode>> landmark_nodes_;

    std::unordered_map<int, boost::shared_ptr<CeresFactor>> factors_;
    std::unordered_map<int, boost::shared_ptr<CeresFactor>> essential_factors_;

    std::deque<SemanticKeyframe::Ptr> kfs_to_process_;
    SemanticKeyframe::Ptr last_kf_processed_;

    bool running_;
    std::thread work_thread_;

    void extractKeypointsThread();

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};