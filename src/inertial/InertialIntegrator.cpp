#include "semantic_slam/inertial/InertialIntegrator.h"

#include <algorithm>

InertialIntegrator::InertialIntegrator()
  : gravity_(0, 0, -9.81)
  , Q_(Eigen::MatrixXd::Zero(15, 15))
{}

InertialIntegrator::InertialIntegrator(const Eigen::Vector3d& gravity)
  : gravity_(gravity)
  , Q_(Eigen::MatrixXd::Zero(15, 15))
{}

void
InertialIntegrator::addData(double t,
                            const Eigen::Vector3d& accel,
                            const Eigen::Vector3d& omega)
{
    imu_times_.push_back(t);
    accels_.push_back(accel);
    omegas_.push_back(omega);
}

void
InertialIntegrator::setAdditiveMeasurementNoise(double gyro_sigma,
                                                double accel_sigma)
{
    Q_.block<3, 3>(0, 0) =
      gyro_sigma * gyro_sigma * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(6, 6) =
      accel_sigma * accel_sigma * Eigen::Matrix3d::Identity();
}

void
InertialIntegrator::setBiasRandomWalkNoise(double gyro_sigma,
                                           double accel_sigma)
{
    Q_.block<3, 3>(3, 3) =
      gyro_sigma * gyro_sigma * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(9, 9) =
      accel_sigma * accel_sigma * Eigen::Matrix3d::Identity();
}

Eigen::Vector3d
InertialIntegrator::interpolateData(double t,
                                    const std::vector<double>& times,
                                    const aligned_vector<Eigen::Vector3d>& data)
{
    // Find first indices before and after t and linearly interpolate the omega

    auto it = std::lower_bound(times.begin(), times.end(), t);

    if (it == times.end()) {
        throw std::runtime_error("Error: not enough data to do interpolation.");
    }

    int idx_end = it - times.begin();

    if (*it == t) {
        return data[idx_end];
    }

    int idx_begin = idx_end - 1;

    double t_offset = t - times[idx_begin];
    double dt = times[idx_end] - times[idx_begin];

    return data[idx_begin] +
           (data[idx_end] - data[idx_begin]) * (t_offset / dt);
}