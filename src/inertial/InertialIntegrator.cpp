#include "semantic_slam/inertial/InertialIntegrator.h"

#include <algorithm>

InertialIntegrator::InertialIntegrator()
  : bias_covariance_(Eigen::MatrixXd::Zero(6, 6))
  , Q_(Eigen::MatrixXd::Zero(6, 6))
  , Q_random_walk_(Eigen::MatrixXd::Zero(6, 6))
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
InertialIntegrator::setAdditiveMeasurementNoise(
  const std::vector<double>& gyro_sigma,
  const std::vector<double>& accel_sigma)
{
    Eigen::VectorXd covs(6);
    covs << gyro_sigma[0], gyro_sigma[1], gyro_sigma[2], accel_sigma[0],
      accel_sigma[1], accel_sigma[2];

    Q_ = covs.array().pow(2).matrix().asDiagonal();
}

void
InertialIntegrator::setBiasRandomWalkNoise(
  const std::vector<double>& gyro_sigma,
  const std::vector<double>& accel_sigma)
{
    Eigen::VectorXd covs(6);
    covs << gyro_sigma[0], gyro_sigma[1], gyro_sigma[2], accel_sigma[0],
      accel_sigma[1], accel_sigma[2];

    Q_random_walk_ = covs.array().pow(2).matrix().asDiagonal();
}

void
InertialIntegrator::setInitialBiasCovariance(const Eigen::MatrixXd& covariance)
{
    bias_covariance_ = covariance;
}

Eigen::Vector3d
InertialIntegrator::interpolateData(double t,
                                    const std::vector<double>& times,
                                    const aligned_vector<Eigen::Vector3d>& data)
{
    // Find first indices before and after t and linearly interpolate the omega

    // std::cout << "interpolating to time " << t << std::endl;

    if (times.size() == 0) {
        std::string msg("Error: not enough data to do interpolation [CODE: 1].");
        throw std::runtime_error(msg);
    }
    
    if (t < times[0]) {
        std::string msg("Error: not enough data to do interpolation [CODE: 2].");
        throw std::runtime_error(msg);
    }

    auto it = std::lower_bound(times.begin(), times.end(), t);

    if (it == times.end()) {
        std::string msg("Error: not enough data to do interpolation [CODE: 3].");
        throw std::runtime_error(msg);
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

Eigen::Vector3d
InertialIntegrator::a_msmt(double t)
{
    return interpolateData(t, imu_times_, accels_);
}

Eigen::Vector3d
InertialIntegrator::averageMeasurement(
  double t1,
  double t2,
  const aligned_vector<Eigen::Vector3d>& data)
{
    Eigen::Vector3d average = Eigen::Vector3d::Zero();
    size_t n_averaged = 0;
    for (size_t i = 0; i < imu_times_.size(); ++i) {
        if (imu_times_[i] < t1)
            continue;

        if (imu_times_[i] > t2)
            break;

        average += data[i];
        n_averaged++;
    }

    return average / n_averaged;
}

Eigen::VectorXd
InertialIntegrator::integrateRK4(
  const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& f,
  double t1,
  double t2,
  const Eigen::VectorXd& y0,
  double step_size)
{
    double t = t1;
    Eigen::VectorXd y = y0;

    while (t + step_size <= t2) {
        y = rk4_iteration(f, t, t + step_size, y);
        t += step_size;
    }

    if (t < t2) {
        y = rk4_iteration(f, t, t2, y);
    }

    return y;
}

Eigen::VectorXd
InertialIntegrator::rk4_iteration(
  const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& f,
  double t1,
  double t2,
  const Eigen::VectorXd& y0)
{
    double h = t2 - t1;
    Eigen::VectorXd k1 = h * f(t1, y0);
    Eigen::VectorXd k2 = h * f(t1 + h / 2.0, y0 + k1 / 2.0);
    Eigen::VectorXd k3 = h * f(t1 + h / 2.0, y0 + k2 / 2.0);
    Eigen::VectorXd k4 = h * f(t1 + h, y0 + k3);
    return y0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

aligned_vector<Eigen::MatrixXd>
InertialIntegrator::integrateRK4(
  const std::function<aligned_vector<
    Eigen::MatrixXd>(double, const aligned_vector<Eigen::MatrixXd>&)>& f,
  double t1,
  double t2,
  const aligned_vector<Eigen::MatrixXd>& x0,
  double step_size)
{
    double t = t1;
    aligned_vector<Eigen::MatrixXd> x;
    for (auto& xval : x0) {
        x.push_back(xval);
    }

    while (t + step_size <= t2) {
        x = rk4_iteration(f, t, t + step_size, x);
        t += step_size;
    }

    if (t < t2) {
        x = rk4_iteration(f, t, t2, x);
    }

    return x;
}

aligned_vector<Eigen::MatrixXd>
InertialIntegrator::rk4_iteration(
  const std::function<aligned_vector<
    Eigen::MatrixXd>(double, const aligned_vector<Eigen::MatrixXd>&)>& f,
  double t1,
  double t2,
  const aligned_vector<Eigen::MatrixXd>& x0)
{
    double h = t2 - t1;

    auto k1_h = f(t1, x0);

    aligned_vector<Eigen::MatrixXd> x1, x2, x3;

    for (size_t i = 0; i < x0.size(); ++i) {
        x1.push_back(x0[i] + h * k1_h[i] / 2);
    }

    auto k2_h = f(t1 + h / 2, x1);
    for (size_t i = 0; i < x0.size(); ++i) {
        x2.push_back(x0[i] + h * k2_h[i] / 2);
    }

    auto k3_h = f(t1 + h / 2, x2);
    for (size_t i = 0; i < x0.size(); ++i) {
        x3.push_back(x0[i] + h * k3_h[i]);
    }

    auto k4_h = f(t1 + h, x3);
    aligned_vector<Eigen::MatrixXd> result;
    for (size_t i = 0; i < x0.size(); ++i) {
        result.push_back(
          x0[i] +
          (h * k1_h[i] + 2 * h * k2_h[i] + 2 * h * k3_h[i] + h * k4_h[i]) / 6);
    }

    return result;
}

// Eigen::MatrixXd
// InertialIntegrator::quaternionMatrixOmega(const Eigen::VectorXd& w)
// {
//     Eigen::MatrixXd Omega(4, 4);

//     // clang-format off
//     Omega <<  0.0,    w(2),  -w(1),   w(0),
//              -w(2),   0.0,    w(0),   w(1),
//               w(1),  -w(0),   0.0,    w(2),
//              -w(0),  -w(1),  -w(2),   0.0;
//     // clang-format on

//     return Omega;
// }

Eigen::MatrixXd
InertialIntegrator::Dqdot_dnoise(const Eigen::VectorXd& q)
{
    Eigen::MatrixXd H(4, 3);

    // clang-format off
    H << -q(3),  q(2), -q(1), 
         -q(2), -q(3),  q(0),
          q(1), -q(0), -q(3), 
          q(0),  q(1),  q(2);
    // clang-format on

    return 0.5 * H;
}

Eigen::MatrixXd
InertialIntegrator::Dbias_sensitivity_dt(double t,
                                         const Eigen::VectorXd& state,
                                         const Eigen::MatrixXd& sensitivities,
                                         const Eigen::VectorXd& gyro_accel_bias)
{
    // D(sensitivity)/dt = sdot = d(qvp_dot)/d(qvp) * s + d(qvp)/d(biases)

    Eigen::MatrixXd Jstate = Dstatedot_dstate(t, state, gyro_accel_bias);
    Eigen::MatrixXd Jbias = Dstatedot_dbias(t, state, gyro_accel_bias);

    // Jstate is 10x10, Jbias is 10x6, sensitivities are 10x6
    return Jstate * sensitivities + Jbias;
}

Eigen::MatrixXd
InertialIntegrator::Dstatedot_dbias(double t,
                                    const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& gyro_accel_bias)
{
    // Resulting matrix here will be 10 by 6, derivative of the qvp state
    // vector derivative with respect to the initial values when starting the
    // preintegration of [bg ba]
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(10, 6);

    Eigen::Quaterniond q(state.head<4>());

    // State vector ordering [q v p]
    // Row indices:          [0 4 7]
    J.block<4, 3>(0, 0) = Dqdot_dnoise(state.head<4>());

    J.block<3, 3>(4, 3) = -q.toRotationMatrix();

    return J;
}

Eigen::MatrixXd
InertialIntegrator::Dstatedot_dstate(double t,
                                     const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& gyro_accel_bias)
{
    // Compute the full Jacobian of the qvp state derivative w.r.t. the
    // qvp state
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(10, 10);

    Eigen::Vector3d a = interpolateData(t, imu_times_, accels_);
    Eigen::Vector3d w = interpolateData(t, imu_times_, omegas_);

    Eigen::Vector4d q_vec = state.head<4>();
    Eigen::Quaterniond q(q_vec);
    q.normalize();

    Eigen::Vector3d bg = gyro_accel_bias.head<3>();
    Eigen::Vector3d ba = gyro_accel_bias.tail<3>();

    // Full state vector ordering is [q v p] indices [0 4 7]
    // Begin with rotation derivatives dq/dx
    // J.block<3, 3>(0, 0) = -skewsymm(w - bg);
    J.block<4, 4>(0, 0) = 0.5 * math::quat_Omega(w - bg);

    J.block<3, 4>(4, 0) = math::Dpoint_transform_dq(q, a - ba);

    // dp/dx
    J.block<3, 3>(7, 4) = Eigen::Matrix3d::Identity();

    return J;
}

Eigen::MatrixXd
InertialIntegrator::Dstatedot_dstate_with_bias(
  double t,
  const Eigen::VectorXd& state,
  const Eigen::VectorXd& gyro_accel_bias)
{
    // Compute full Jacobian of d([q v p bg ba])/dt w.r.t. full state

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(16, 16);

    J.topLeftCorner<10, 10>() = Dstatedot_dstate(t, state, gyro_accel_bias);
    J.topRightCorner<10, 6>() = Dstatedot_dbias(t, state, gyro_accel_bias);

    // dbg = dba = 0

    return J;
}

Eigen::MatrixXd
InertialIntegrator::Dstatedot_dnoise(double t,
                                     const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& gyro_accel_bias)
{
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(10, 6);

    // Noise vector ordering here is
    // [nr na] where nr is additive gyro noise, na is additive accel noise
    // note: we are IGNORING the effect of the gyro bias random walk
    // within an integration period, TODO fix this?

    Eigen::Quaterniond q(state.head<4>());

    G.block<4, 3>(0, 0) = Dqdot_dnoise(state.head<4>());
    G.block<3, 3>(4, 3) = -q.toRotationMatrix();

    return G;
}

Eigen::VectorXd
InertialIntegrator::statedot_preint(double t,
                                    const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& gyro_accel_bias)
{
    Eigen::VectorXd xdot(10, 1);

    Eigen::Vector4d quat = state.head<4>();

    xdot.head<4>() =
      0.5 *
      math::quat_Omega(interpolateData(t, imu_times_, omegas_) -
                            gyro_accel_bias.head<3>()) *
      quat;

    quat.normalize();
    if (quat(3) < 0.0)
        quat = -quat;

    // Velocity derivative
    Eigen::Quaterniond q(quat);

    xdot.segment<3>(4) =
      q.toRotationMatrix() *
      (interpolateData(t, imu_times_, accels_) - gyro_accel_bias.tail<3>());

    // Position derivative
    xdot.tail<3>() = state.segment<3>(4);

    return xdot;
}

Eigen::VectorXd
InertialIntegrator::statedot(double t,
                             const Eigen::VectorXd& state,
                             const Eigen::VectorXd& gyro_accel_bias,
                             const Eigen::VectorXd& gravity)
{
    Eigen::VectorXd xdot(10, 1);

    Eigen::Matrix<double, 4, 1> quat = state.head<4>();

    xdot.head<4>() =
      0.5 *
      math::quat_Omega(interpolateData(t, imu_times_, omegas_) -
                            gyro_accel_bias.head<3>()) *
      quat;

    quat.normalize();
    if (quat(3) < 0.0)
        quat = -quat;

    // Velocity derivative
    Eigen::Quaterniond q(quat);

    xdot.segment<3>(4) =
      q.toRotationMatrix() *
        (interpolateData(t, imu_times_, accels_) - gyro_accel_bias.tail<3>()) +
      gravity;

    // Position derivative
    xdot.tail<3>() = state.segment<3>(4);

    return xdot;
}

Eigen::MatrixXd
InertialIntegrator::Pdot(double t,
                         const Eigen::VectorXd& state,
                         const Eigen::MatrixXd& P,
                         const Eigen::VectorXd& gyro_accel_bias)
{
    // Use continuous time kalman filter equation:
    //   Pdot = F*P + P*F' + G*Q*G'
    // First compute error state transition matrix F and noise matrix G
    // Eigen::MatrixXd F_full = Eigen::MatrixXd::Zero(16, 16);
    // F_full.topLeftCorner<10, 10>() =
    //   Dstatedot_dstate(t, state, gyro_accel_bias);
    // F_full.topRightCorner<10, 6>() = Dstatedot_dbias(t, state, gyro_accel_bias);

    // Eigen::MatrixXd G_full = Dstatedot_dnoise(t, state, gyro_accel_bias);

    // // Need to translate F_full from the full ambient quaternion space to
    // // the local space.
    // Eigen::Matrix<double, 4, 3, Eigen::RowMajor> Dqfull_dqlocal;
    // QuaternionLocalParameterization().ComputeJacobian(state.head<4>().data(),
    //                                                   Dqfull_dqlocal.data());

    // Eigen::MatrixXd F(15, 15);
    // F.block<3, 3>(0, 0) =
    //   2 * (F_full.block<4, 4>(0, 0) * Dqfull_dqlocal).topRows<3>();

    // F.block<3, 12>(0, 3) = 2 * F_full.block<3, 12>(0, 4);

    // F.block<12, 3>(3, 0) = F_full.block<12, 4>(4, 0) * Dqfull_dqlocal;

    // F.block<12, 12>(3, 3) = F_full.block<12, 12>(4, 4);

    // Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 6);
    // G.topRows<3>() = 2 * G_full.topRows<3>();
    // G.block<6, 6>(3, 0) = G_full.bottomRows<6>();

    // Eigen::MatrixXd G_walk = Eigen::MatrixXd::Zero(15, 6);
    // G_walk.bottomRows<6>() = Eigen::MatrixXd::Identity(6, 6);

    // return F * P + P * F.transpose() + G * Q_ * G.transpose() +
    //        G_walk * Q_random_walk_ * G_walk.transpose();

    auto a = interpolateData(t, imu_times_, accels_);
    auto w = interpolateData(t, imu_times_, omegas_);

    auto bw = gyro_accel_bias.head<3>();
    auto ba = gyro_accel_bias.tail<3>();

    Eigen::Vector4d q_vec = state.head<4>();
    Eigen::Quaterniond q(q_vec);
    q.normalize();

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 6);
    Eigen::MatrixXd G_walk = Eigen::MatrixXd::Zero(15, 6);

    // dq/dq
    F.block<3, 3>(0, 0) = -math::skewsymm(w - bw);

    // dq/d(bw)
    F.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();

    // dv/dq
    F.block<3, 3>(3, 0) = -q.toRotationMatrix() * math::skewsymm(a - ba);

    // dv/d(ba)
    F.block<3, 3>(3, 12) = -q.toRotationMatrix();

    // dp/dv
    F.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();

    // dq/d(additive gyro noise)
    G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();

    // da/d(additive accel noise)
    G.block<3, 3>(3, 3) = -q.toRotationMatrix();

    // d(bg)/d(gyro walk noise)
    G_walk.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();

    // d(ba)/d(accel walk noise)
    G_walk.block<3, 3>(12, 3) = Eigen::Matrix3d::Identity();



    return F * P + P * F.transpose() + G * Q_ * G.transpose() +
           G_walk * Q_random_walk_ * G_walk.transpose();

}

Eigen::VectorXd
InertialIntegrator::integrateInertial(double t1,
                                      double t2,
                                      const Eigen::VectorXd& qvp0,
                                      const Eigen::VectorXd& gyro_accel_bias,
                                      const Eigen::VectorXd& gravity)
{
    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> f =
      [&](double t, const Eigen::VectorXd& x) {
          return this->statedot(t, x, gyro_accel_bias, gravity);
      };

    Eigen::VectorXd qvp_plus = this->integrateRK4(f, t1, t2, qvp0, 0.005);

    qvp_plus.head<4>() /= qvp_plus.head<4>().norm();

    if (qvp_plus(3) < 0) {
      qvp_plus.head<4>() = -qvp_plus.head<4>();
    }

    return qvp_plus;
}

Eigen::VectorXd
InertialIntegrator::preintegrateInertial(double t1,
                                         double t2,
                                         const Eigen::VectorXd& gyro_accel_bias)
{
    Eigen::VectorXd qvp_identity(10);
    qvp_identity.setZero();
    qvp_identity(3) = 1.0;

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> f =
      [&](double t, const Eigen::VectorXd& x) {
          return this->statedot_preint(t, x, gyro_accel_bias);
      };

    Eigen::VectorXd x = this->integrateRK4(f, t1, t2, qvp_identity, 0.005);
    x.head<4>() /= x.head<4>().norm();
    if (x(3) < 0) {
        x.head<4>() *= -1;
    }

    return x;
}

aligned_vector<Eigen::MatrixXd>
InertialIntegrator::integrateInertialWithCovariance(
  double t1,
  double t2,
  const Eigen::VectorXd& qvp0,
  const Eigen::VectorXd& gyro_accel_bias,
  const Eigen::VectorXd& gravity)
{
    std::function<aligned_vector<Eigen::MatrixXd>(
      double, const aligned_vector<Eigen::MatrixXd>&)>
      f = [&](double t, const aligned_vector<Eigen::MatrixXd>& xP) {
          return aligned_vector<Eigen::MatrixXd>{
              this->statedot(t, xP[0], gyro_accel_bias, gravity),
              this->Pdot(t, xP[0], xP[1], gyro_accel_bias)
          };
      };

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(15, 15);

    auto xP = this->integrateRK4(f, t1, t2, { qvp0, P0 }, 0.005);
    Eigen::VectorXd x = xP[0];
    xP[0].topRows<4>() /= x.head<4>().norm();
    if (x(3) < 0) {
        xP[0].topRows<4>() *= -1;
    }

    return xP;
}

aligned_vector<Eigen::MatrixXd>
InertialIntegrator::preintegrateInertialWithJacobianAndCovariance(
  double t1,
  double t2,
  const Eigen::VectorXd& gyro_accel_bias)
{
    std::function<aligned_vector<Eigen::MatrixXd>(
      double, const aligned_vector<Eigen::MatrixXd>&)>
      f = [&](double t, const aligned_vector<Eigen::MatrixXd>& xJP) {
          return aligned_vector<Eigen::MatrixXd>{
              this->statedot_preint(t, xJP[0], gyro_accel_bias),
              this->Dbias_sensitivity_dt(t, xJP[0], xJP[1], gyro_accel_bias),
              this->Pdot(t, xJP[0], xJP[2], gyro_accel_bias)
          };
      };

    Eigen::VectorXd qvp_identity(10);
    qvp_identity.setZero();
    qvp_identity(3) = 1.0;

    Eigen::MatrixXd Jbias0 = Eigen::MatrixXd::Zero(10, 6);
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(15, 15);

    auto xJP =
      this->integrateRK4(f, t1, t2, { qvp_identity, Jbias0, P0 }, 0.005);

    Eigen::VectorXd x = xJP[0];
    xJP[0].topRows<4>() /= x.head<4>().norm();
    if (x(3) < 0) {
        xJP[0].topRows<4>() *= -1;
    }

    // J here is d(qvp)/d(bias0).
    // To account for initial bias covariance we need d(q v p bg ba)/d(bias0).
    Eigen::MatrixXd Jbias_full = Eigen::MatrixXd(16, 6);
    Jbias_full.topRows<10>() = xJP[1];
    Jbias_full.bottomRows<6>() = Eigen::MatrixXd::Identity(6, 6);

    // The initial bias covariance doesn't affect the integration of the
    // covariance but it does effect the final value through the bias jacobian.
    // Unfortunately Jbias is w.r.t. the ambient quaternion space and P is
    // w.r.t. the tangent space so we need another mapping here
    Eigen::MatrixXd amb_to_tangent = Eigen::MatrixXd::Zero(15, 16);
    amb_to_tangent.topLeftCorner<3, 3>() = 2.0 * Eigen::Matrix3d::Identity();
    amb_to_tangent.bottomRightCorner<12, 12>() =
      Eigen::MatrixXd::Identity(12, 12);

    xJP[2] += (amb_to_tangent * Jbias_full) * bias_covariance_ *
              (amb_to_tangent * Jbias_full).transpose();

    return xJP;
}

boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
InertialIntegrator::createGtsamParams()
{
    auto params =
      gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);

    params->accelerometerCovariance = Q_.bottomRightCorner<3, 3>();
    params->gyroscopeCovariance = Q_.topLeftCorner<3, 3>();
    params->biasOmegaCovariance = Q_random_walk_.topLeftCorner<3, 3>();
    params->biasAccCovariance = Q_random_walk_.bottomRightCorner<3, 3>();

    params->integrationCovariance = 1e-8 * Eigen::MatrixXd::Identity(3, 3);

    params->biasAccOmegaInt = Eigen::MatrixXd::Zero(6, 6);
    params->biasAccOmegaInt.topLeftCorner<3, 3>() =
      bias_covariance_.bottomRightCorner<3, 3>();
    params->biasAccOmegaInt.bottomRightCorner<3, 3>() =
      bias_covariance_.topLeftCorner<3, 3>();
    params->biasAccOmegaInt.bottomLeftCorner<3, 3>() =
      bias_covariance_.topRightCorner<3, 3>();
    params->biasAccOmegaInt.topRightCorner<3, 3>() =
      bias_covariance_.bottomLeftCorner<3, 3>();

    return params;
}

boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements>
InertialIntegrator::createGtsamIntegrator(double t0,
                                          double t1,
                                          const Eigen::VectorXd& bias0)
{
    auto gtsam_params = createGtsamParams();
    // gtsam bias ordering is [a, w]; ours is [w, a]
    gtsam::imuBias::ConstantBias gtsam_bias(bias0.tail<3>(), bias0.head<3>());

    auto gtsam_preintegrator =
      boost::make_shared<gtsam::PreintegratedCombinedMeasurements>(gtsam_params,
                                                                   gtsam_bias);

    // Iterate through the desired times and measurements and add them to the
    // GTSAM integrator

    auto start_it = std::lower_bound(imu_times_.begin(), imu_times_.end(), t0);
    size_t idx = start_it - imu_times_.begin();
    double t = *start_it;
    double next_t = t;

    while (idx < imu_times_.size() - 1 && next_t < t1) {
        t = imu_times_[idx];
        next_t = imu_times_[idx + 1];
        double dt = next_t - t;

        gtsam_preintegrator->integrateMeasurement(
          accels_[idx], omegas_[idx], dt);

        idx++;
    }

    if (next_t < t1) {
        throw std::runtime_error("Not enough data to perform preintegration!");
    }

    return gtsam_preintegrator;
}