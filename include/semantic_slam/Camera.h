#pragma once

#include "semantic_slam/CameraCalibration.h"

class Camera {
public:
    Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration = nullptr);

    Eigen::Vector2d project(const Eigen::Vector3d& p, 
                            boost::optional<Eigen::MatrixXd&> Hpose = boost::none, 
                            boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    Eigen::Vector2d calibrate(const Eigen::Vector2d& p) const; 
    Eigen::Vector2d uncalibrate(const Eigen::Vector2d& p) const; 

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

Camera::Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration)
    : pose_(pose), calibration_(calibration)
{

}

Eigen::Vector2d 
Camera::project(const Eigen::Vector3d& p, 
                boost::optional<Eigen::MatrixXd&> Hpose, 
                boost::optional<Eigen::MatrixXd&> Hpoint) const
{
    // Convert to camera coordinates
    Eigen::MatrixXd Htransform_pose, Htransform_point;
    Eigen::Vector3d C_p;

    if (Hpose || Hpoint) C_p = pose_.transform_to(p, Htransform_pose, Htransform_point);
    else C_p = pose_.transform_to(p);

    if (C_p(2) <= 0) {
        throw CheiralityException("Cheirality exception");
    }

    // Project to normalized coordinates
    double zinv = 1.0 / C_p(2);
    Eigen::Vector2d xy(zinv * C_p(0),
                       zinv * C_p(1));

    Eigen::Matrix<double, 2, 3> Hproject;

    if (Hpose || Hpoint) {
        Hproject << 1 , 0 , -C_p(0)/C_p(2),
                    0 , 1 , -C_p(1)/C_p(2);
        Hproject /= C_p(2);
    }

    // Eigen::Matrix<double, 2, 3> Dproject;
    // Dproject << zinv, 0.0, -C_p(0) * zinv, 0.0, zinv, -C_p(1) * zinv;

    // Distort if we have a calibration
    Eigen::MatrixXd Hcalib = Eigen::MatrixXd::Identity(2,2);

    if (calibration_) {
        if (Hpose || Hpoint) xy = calibration_->uncalibrate(xy, Hcalib);
        else xy = calibration_->uncalibrate(xy);
    }

    if (Hpose) {
        *Hpose = Hcalib * Hproject * Htransform_pose;
    }

    if (Hpoint) {
        *Hpoint = Hcalib * Hproject * Htransform_point;
    }

    return xy;
}



Eigen::Vector2d Camera::calibrate(const Eigen::Vector2d& p) const 
{ 
    if (calibration_) return calibration_->calibrate(p); 
    else return p;
}

Eigen::Vector2d Camera::uncalibrate(const Eigen::Vector2d& p) const 
{ 
    if (calibration_) return calibration_->uncalibrate(p); 
    else return p;
}