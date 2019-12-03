#pragma once

#include "semantic_slam/Common.h"

class CameraCalibration;

class Camera {
public:
    Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration = nullptr);

    Eigen::Vector2d project(const Eigen::Vector3d& p, 
                            boost::optional<Eigen::MatrixXd&> Hpose = boost::none, 
                            boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    Eigen::Vector2d calibrate(const Eigen::Vector2d& p) const; 
    Eigen::Vector2d uncalibrate(const Eigen::Vector2d& p) const; 

    boost::shared_ptr<CameraCalibration> calibration() const { return calibration_; }

    Pose3& pose() { return pose_; }
    const Pose3& pose() const { return pose_; }

private:
    Pose3 pose_;
    boost::shared_ptr<CameraCalibration> calibration_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class CheiralityException : public std::exception
{
public:
    CheiralityException(std::string message) : msg_(message) { }
    const char* what() { return msg_.c_str(); }
private:
    std::string msg_;
};
