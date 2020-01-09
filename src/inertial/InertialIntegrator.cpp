#include "semantic_slam/inertial/InertialIntegrator.h"

InertialIntegrator::InertialIntegrator()
  : gravity_(0, 0, -9.81)
{}

InertialIntegrator::InertialIntegrator(const Eigen::Vector3d& gravity)
  : gravity_(gravity)
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

Eigen::Vector3d
InertialIntegrator::interpolateData(double t,
                                    const std::vector<double>& times,
                                    const aligned_vector<Eigen::Vector3d>& data)
{
    // Find first indices before and after t and linearly interpolate the omega
    // values
    int idx_begin = -1;
    int idx_end = -1;
    for (size_t i = 0; i < times.size(); ++i) {
        if (times[i] <= t) {
            idx_begin = i;
        }

        if (times[i] >= t) {
            idx_end = i;
            break;
        }
    }

    if (idx_begin == -1 || idx_end == -1) {
        throw std::runtime_error("Error: not enough data to do interpolation.");
    }

    if (idx_begin == idx_end) {
        return data[idx_begin];
    }

    double t_offset = t - times[idx_begin];
    double dt = times[idx_end] - times[idx_begin];

    return data[idx_begin] +
           (data[idx_end] - data[idx_begin]) * (t_offset / dt);
}