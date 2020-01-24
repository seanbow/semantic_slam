#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/Pose3.h"

#include <deque>
#include <mutex>
#include <sensor_msgs/Imu.h>
#include <unordered_map>
// #include <gtsam/geometry/Pose3.h>

class InertialIntegrator;

class InertialOdometryHandler : public OdometryHandler
{
  public:
    void setup();

    void msgCallback(const sensor_msgs::Imu::ConstPtr& msg);

    boost::shared_ptr<SemanticKeyframe> originKeyframe();

    boost::shared_ptr<SemanticKeyframe> createKeyframe(ros::Time time);

    boost::shared_ptr<SemanticKeyframe> findNearestKeyframe(ros::Time time);

    bool getRelativePoseEstimateTo(ros::Time t, Pose3& T12);

    void updateKeyframeAfterOptimization(
      boost::shared_ptr<SemanticKeyframe> keyframe_to_update,
      boost::shared_ptr<SemanticKeyframe> optimized_keyframe);

    // inherit constructor
    using OdometryHandler::OdometryHandler;

  private:
    ros::Subscriber subscriber_;

    std::deque<sensor_msgs::Imu> msg_queue_;

    std::vector<boost::shared_ptr<SemanticKeyframe>> keyframes_;

    boost::shared_ptr<InertialIntegrator> integrator_;

    std::mutex mutex_;

    Pose3 last_odom_;
    ros::Time last_time_;
    size_t last_keyframe_index_;

    size_t received_msgs_;
    size_t last_msg_seq_;

    unsigned char node_chr_;

    std::vector<double> a_bias_init_;
    std::vector<double> w_bias_init_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
