#include "semantic_slam/InertialOdometryHandler.h"

#include "semantic_slam/CeresImuFactor.h"
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/ImuBiasNode.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/inertial/InertialIntegrator.h"

#include <string>

using namespace std::string_literals;
namespace sym = symbol_shorthand;

void
InertialOdometryHandler::setup()
{
    ROS_INFO("Starting IMU handler.");
    std::string imu_topic;
    pnh_.param("imu_topic", imu_topic, "/imu0"s);

    ROS_INFO_STREAM("Subscribing to topic " << imu_topic);

    subscriber_ = nh_.subscribe(
      imu_topic, 10000, &InertialOdometryHandler::msgCallback, this);

    received_msgs_ = 0;
    last_keyframe_index_ = 0;
    last_msg_time_ = 0;

    node_chr_ = 'x';

    integrator_ = util::allocate_aligned<InertialIntegrator>();

    // Read IMU calibration parameters
    std::vector<double> a_sigma, w_sigma;
    std::vector<double> a_bias_sigma, w_bias_sigma;

    if (!pnh_.getParam("a_sigma", a_sigma) ||
        !pnh_.getParam("w_sigma", w_sigma) ||
        !pnh_.getParam("a_bias_sigma", a_bias_sigma) ||
        !pnh_.getParam("w_bias_sigma", w_bias_sigma)) {
        ROS_ERROR_STREAM("Unable to read IMU calibration parameters");
    }

    integrator_->setAdditiveMeasurementNoise(w_sigma, a_sigma);
    integrator_->setBiasRandomWalkNoise(w_bias_sigma, a_bias_sigma);

    // Read initial bias values

    if (!pnh_.getParam("a_bias_init", a_bias_init_) ||
        !pnh_.getParam("w_bias_init", w_bias_init_)) {
        ROS_ERROR_STREAM("Unable to read initial IMU bias values");

        a_bias_init_.resize(3, 0.0);
        w_bias_init_.resize(3, 0.0);
    }
}

void
InertialOdometryHandler::msgCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    received_msgs_++;

    if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM("[IMU Handler] Error: dropped message. Expected "
                         << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        msg_queue_.push_back(*msg);

        last_msg_seq_ = msg->header.seq;
        last_msg_time_ = msg->header.stamp.toSec();

        // For now, just have one single integrator with all of our data.
        Eigen::Vector3d a, w;
        a << msg->linear_acceleration.x, msg->linear_acceleration.y,
          msg->linear_acceleration.z;
        w << msg->angular_velocity.x, msg->angular_velocity.y,
          msg->angular_velocity.z;

        integrator_->addData(msg->header.stamp.toSec(), a, w);
    }

    cv_.notify_all();
}

bool
InertialOdometryHandler::getRelativePoseEstimateTo(ros::Time t, Pose3& T12)
{
    if (keyframes_.empty() || t.toSec() > integrator_->latestTime())
        return false;

    // Start integrating from identity rotation & translation but with
    // our actual current velocity estimate

    Eigen::VectorXd qvp = Eigen::VectorXd::Zero(10);
    qvp(3) = 1.0;

    qvp.segment<3>(4) = keyframes_.back()->velocity();

    auto bias = keyframes_.back()->bias();

    auto dxhat =
      integrator_->integrateInertial(keyframes_.back()->time().toSec(),
                                     t.toSec(),
                                     qvp,
                                     bias,
                                     mapper_->gravity());

    T12 = Pose3(Eigen::Quaterniond(dxhat.head<4>()), dxhat.tail<3>());

    return true;

    // auto bias_node = dx = integrator_->integrateInertial
}

SemanticKeyframe::Ptr
InertialOdometryHandler::createKeyframe(ros::Time time)
{
    // Integrate up to time and attach a node corresponding to it

    Symbol keyframe_symbol(node_chr_, last_keyframe_index_ + 1);

    if (keyframes_.empty()) {
        // This is the first keyframe -- create the origin frame
        throw std::logic_error(
          "createKeyframe should only be called AFTER initializing an origin "
          "frame with originKeyframe!");
        // SemanticKeyframe::Ptr frame = originKeyframe();
        // return frame;
    }

    auto last_kf = keyframes_.back();
    Eigen::VectorXd qvp0(10);
    qvp0.head<4>() = last_kf->pose().rotation().coeffs();
    qvp0.segment<3>(4) = last_kf->velocity();
    qvp0.tail<3>() = last_kf->pose().translation();

    integrator_->setInitialBiasCovariance(last_kf->bias_covariance());

    Eigen::VectorXd xhat;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&](){
              if (last_msg_time_ < time.toSec()) {
                  // ROS_WARN_STREAM("IMU data is " << time.toSec() - last_msg_time_ << " seconds behind. Waiting...");
                  return false;
              } else {
                  return true;
              }; 
            });

        xhat = integrator_->integrateInertial(last_kf->time().toSec(),
                                              time.toSec(),
                                              qvp0,
                                              last_kf->bias(),
                                              mapper_->gravity());
    }

    Pose3 G_T_x(Eigen::Quaterniond(xhat.head<4>()), xhat.tail<3>());

    auto new_kf =
      util::allocate_aligned<SemanticKeyframe>(keyframe_symbol, time, true);
    keyframes_.push_back(new_kf);

    new_kf->pose() = G_T_x;
    new_kf->velocity() = xhat.segment<3>(4);
    new_kf->bias() = last_kf->bias();

    new_kf->graph_node()->pose() = new_kf->pose();
    new_kf->velocity_node()->vector() = new_kf->velocity();
    new_kf->bias_node()->vector() = new_kf->bias();

    new_kf->odometry() = G_T_x; // does this make sense?
    new_kf->image_time = time;

    auto imu_fac =
      util::allocate_aligned<CeresImuFactor>(last_kf->graph_node(),
                                             last_kf->velocity_node(),
                                             last_kf->bias_node(),
                                             new_kf->graph_node(),
                                             new_kf->velocity_node(),
                                             new_kf->bias_node(),
                                             mapper_->gravity_node(),
                                             integrator_,
                                             last_kf->time().toSec(),
                                             time.toSec());

    new_kf->spine_factor() = imu_fac;

    last_keyframe_index_++;

    // ROS_INFO_STREAM("Created keyframe with time " << keyframe->time() << "
    // and image time " << keyframe->image_time);

    return new_kf;
}

void
InertialOdometryHandler::updateKeyframeAfterOptimization(
  SemanticKeyframe::Ptr keyframe_to_update,
  SemanticKeyframe::Ptr optimized_keyframe)
{
    // Easiest to just redo the integration
    Eigen::VectorXd qvp(10);
    qvp.head<4>() = optimized_keyframe->pose().rotation().coeffs();
    qvp.segment<3>(4) = optimized_keyframe->velocity();
    qvp.tail<3>() = optimized_keyframe->pose().translation();

    auto xhat =
      integrator_->integrateInertial(optimized_keyframe->time().toSec(),
                                     keyframe_to_update->time().toSec(),
                                     qvp,
                                     optimized_keyframe->bias(),
                                     mapper_->gravity());

    // std::cout
    //   << "Delta velocity = "
    //   << (qvp.segment<3>(4) - keyframe_to_update->velocity()).transpose()
    //   << std::endl;

    keyframe_to_update->pose() =
      Pose3(Eigen::Quaterniond(xhat.head<4>()), xhat.tail<3>());
    keyframe_to_update->velocity() = xhat.segment<3>(4);
    keyframe_to_update->bias() = optimized_keyframe->bias();
}

SemanticKeyframe::Ptr
InertialOdometryHandler::originKeyframe()
{
    while (integrator_->earliestTime() < 0) {
        ros::Duration(0.002).sleep();
    }

    double averaging_time = 0.25;

    while (integrator_->earliestTime() + averaging_time >
           integrator_->latestTime()) {
        ros::Duration(0.002).sleep();
    }

    ros::Time time(integrator_->earliestTime());

    SemanticKeyframe::Ptr kf = util::allocate_aligned<SemanticKeyframe>(
      Symbol(node_chr_, 0), time, true);

    // Estimate the initial orientation based on the first received
    // accelerometer measurement and gravity
    Eigen::Vector3d a = integrator_->a_msmt(integrator_->earliestTime());
    Eigen::Vector3d avg_a = integrator_->averageAcceleration(
      integrator_->earliestTime(),
      integrator_->earliestTime() + averaging_time);
    Eigen::Vector3d avg_w =
      integrator_->averageOmega(integrator_->earliestTime(),
                                integrator_->earliestTime() + averaging_time);

    Eigen::Quaterniond q;
    q.setFromTwoVectors(-avg_a, mapper_->gravity());

    // Guess the initial biases as well assuming zero motion
    Eigen::Vector3d local_gravity = q.conjugate() * mapper_->gravity();
    kf->bias().head<3>() = avg_w;
    kf->bias().tail<3>() = avg_a + local_gravity;

    std::cout << "First accel = " << a.transpose() << std::endl;
    std::cout << "Average accel = " << avg_a.transpose() << std::endl;
    std::cout << "Gravity = " << mapper_->gravity().transpose() << std::endl;
    std::cout << "Initial orientation = " << q.coeffs().transpose()
              << std::endl;

    kf->odometry() = Pose3(q, Eigen::Vector3d::Zero());
    kf->pose() = Pose3(q, Eigen::Vector3d::Zero());
    kf->velocity() = Eigen::Vector3d::Zero();

    // kf->bias().head<3>() = Eigen::Vector3d(w_bias_init_.data());
    // kf->bias().tail<3>() = Eigen::Vector3d(a_bias_init_.data());

    kf->graph_node()->pose() = Pose3::Identity();
    kf->velocity_node()->vector() = kf->velocity();
    kf->bias_node()->vector() = kf->bias();

    keyframes_.push_back(kf);

    return kf;
}