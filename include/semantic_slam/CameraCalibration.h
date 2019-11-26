#pragma once

#include <eigen3/Eigen/Core>
#include <boost/optional.hpp>

// Camera calibration data & utilities

class CameraCalibration {
public:
    CameraCalibration() { }

    CameraCalibration(double fx, double fy, double s, double u0, double v0, double k1, 
            double k2, double p1 = 0.0, double p2 = 0.0);

private:
    double fx_, fy_, s_, u0_, v0_;
    double k1_, k2_; // radial distortion
    double p1_, p2_; // tangential distortion

public:
    inline double fx() const { return fx_; }
    inline double fy() const { return fy_; }
    inline double skew() const { return s_; }
    inline double u0() const { return u0_; }
    inline double v0() const { return v0_; }
    inline double k1() const { return k1_; }
    inline double k2() const { return k2_; }
    inline double p1() const { return p1_; }
    inline double p2() const { return p2_; }

    // Convert intrinsic camera coordinates to distorted image coordinates
    Eigen::Vector2d uncalibrate(const Eigen::Vector2d& p,
                                boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    void Duncalibrate(const Eigen::Vector2d& p, Eigen::MatrixXd& Hpoint) const;

    // Convert distorted image coordinates to intrinsic coordinates
    Eigen::Vector2d calibrate(const Eigen::Vector2d& p) const;
};
