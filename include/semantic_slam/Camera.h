#pragma once

#include "semantic_slam/CameraCalibration.h"

class Camera {
public:
    Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration = nullptr);

    Eigen::Vector2d project(const Eigen::Vector3d& p) const;

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

Camera::Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration)
    : pose_(pose), calibration_(calibration)
{

}

Eigen::Vector2d 
Camera::project(const Eigen::Vector3d& p) const
{
    // Convert to camera coordinates
    Eigen::Vector3d C_p = pose_.transform_to(p);

    if (C_p(2) < 0) {
        throw CheiralityException("Cheirality exception");
    }

    // Project to normalized coordinates
    double zinv = 1.0 / C_p(2);
    Eigen::Vector2d xy(zinv * C_p(0),
                       zinv * C_p(1));

    // Eigen::Matrix<double, 2, 3> Dproject;
    // Dproject << zinv, 0.0, -C_p(0) * zinv, 0.0, zinv, -C_p(1) * zinv;

    // Distort if we have a calibration
    if (calibration_) {
        return calibration_->uncalibrate(xy);
    } else {
        return xy;
    }
}