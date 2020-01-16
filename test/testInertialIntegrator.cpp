

#include "semantic_slam/inertial/InertialIntegrator.h"

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include <ros/package.h>
#include <rosfmt/rosfmt.h>

#include "semantic_slam/CeresImuFactor.h"
#include "semantic_slam/FactorGraph.h"

Eigen::Matrix<double, 6, 1> zero_bias = Eigen::Matrix<double, 6, 1>::Zero();
Eigen::Vector3d G_gravity(0, 0, -9.81);

Eigen::MatrixXd
dlmread(std::string filename)
{
    std::ifstream f(filename);

    // count the number of rows/cols
    std::string line;
    int rows = 0;

    if (!std::getline(f, line)) {
        return Eigen::MatrixXd(0, 0);
    }

    rows++;

    // count cols in first line
    double x;
    int cols = 0;
    std::stringstream stream(line);
    while (stream >> x) {
        cols++;
    }

    // continue until we reach eof
    while (std::getline(f, line)) {
        rows++;
    }

    // Allocate actual matrix, rewind file, actually read data...
    Eigen::MatrixXd data(rows, cols);
    f.clear();
    f.seekg(0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            f >> data(i, j);
        }
    }

    return data;
}

// TEST(InertialIntegratorTest, testRK4_basic)
// {
//     InertialIntegrator integrator;

//     std::function<double(double, const double&)> f =
//       [](double t, const double& x) -> double { return x; };

//     // integration of xdot = f(t,x) = x from t = 0 to t = T should be e^2...

//     double e2 = integrator.integrateRK4(f, 0, 2, 1.0);

//     EXPECT_NEAR(e2, std::exp(2), 1e-4);
// }

TEST(InertialIntegratorTest, testRK4_vector)
{
    InertialIntegrator integrator;

    Eigen::Matrix2d A;
    A << 1, 2, 3, 4;

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> f =
      [&](double t, const Eigen::VectorXd& x) { return A * x; };

    Eigen::VectorXd x0(2);
    x0 << 1, 1;

    Eigen::VectorXd x = integrator.integrateRK4(f, 0, 1, x0, 0.02);

    EXPECT_NEAR(x(0), 126.7055, 1e-2);
    EXPECT_NEAR(x(1), 276.1786, 1e-2);
}

TEST(InertialIntegratorTest, testRK4_inertialStateNoMovement)
{
    InertialIntegrator integrator;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0(3) = 1.0; // set identity quaternion

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x, zero_bias, G_gravity);
      };

    // Some fake data...
    // Zero movement
    Eigen::Vector3d a(0, 0, 9.81);
    Eigen::Vector3d w(0, 0, 0);

    integrator.addData(0, a, w);
    integrator.addData(0.5, a, w);
    integrator.addData(1.0, a, w);
    integrator.addData(1.5, a, w);

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 1.25, x0);

    EXPECT_NEAR(x.norm(), 1.0, 1e-3);
    EXPECT_NEAR(x(3), 1, 1e-3);
    EXPECT_NEAR(x(7), 0, 1e-3);
    EXPECT_NEAR(x(8), 0, 1e-3);
    EXPECT_NEAR(x(9), 0, 1e-3);
}

TEST(InertialIntegratorTest, testRK4_inertialStateNoMovementRotatedFrame)
{
    InertialIntegrator integrator;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);

    // Set initial orientation with y down
    x0(0) = -0.5;
    x0(1) = 0.5;
    x0(2) = -0.5;
    x0(3) = 0.5;

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x, zero_bias, G_gravity);
      };

    // Some fake data...
    // Zero movement
    Eigen::Vector3d a(0, -9.81, 0);
    Eigen::Vector3d w(0, 0, 0);

    integrator.addData(0, a, w);
    integrator.addData(0.5, a, w);
    integrator.addData(1.0, a, w);
    integrator.addData(1.5, a, w);

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 1.25, x0);

    EXPECT_NEAR(x.norm(), 1.0, 1e-3);
    EXPECT_NEAR(x(0), -0.5, 1e-3);
    EXPECT_NEAR(x(7), 0, 1e-3);
    EXPECT_NEAR(x(8), 0, 1e-3);
    EXPECT_NEAR(x(9), 0, 1e-3);
}

TEST(InertialIntegratorTest, testRK4_inertialStateSimpleAccel)
{
    InertialIntegrator integrator;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0(3) = 1.0; // set identity quaternion

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x, zero_bias, G_gravity);
      };

    // Some fake data...
    // simple accel in z direction
    Eigen::Vector3d a(0, 0, 10.81);
    Eigen::Vector3d w(0, 0, 0);

    integrator.addData(0, a, w);
    integrator.addData(0.5, a, w);
    integrator.addData(1.0, a, w);
    integrator.addData(1.5, a, w);

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 1.25, x0);

    // Expect no rotation, v = at, x = .5*a*t^2
    EXPECT_NEAR(x(3), 1.0, 1e-5);
    EXPECT_NEAR(x(6), 1.25, 1e-5);
    EXPECT_NEAR(x(9), 0.5 * 1.25 * 1.25, 1e-5);
}

TEST(InertialIntegratorTest, testRK4_inertialStateSimpleAccelAndSlowdown)
{
    InertialIntegrator integrator;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0(3) = 1.0; // set identity quaternion

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x, zero_bias, G_gravity);
      };

    // Some fake data...
    // simple accel in z direction
    Eigen::Vector3d a(0, 0, 10.81);
    Eigen::Vector3d w(0, 0, 0);

    integrator.addData(0, a, w);
    integrator.addData(0.5, a, w);
    integrator.addData(1.0, a, w);
    integrator.addData(1.5, a, w);

    // slowdown!
    Eigen::Vector3d a2(0, 0, 8.81);
    integrator.addData(2.0, a2, w);
    integrator.addData(2.5, a2, w);
    integrator.addData(3.0, a2, w);
    integrator.addData(3.5, a2, w);

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 3.5, x0);

    // Because we're linearly interpolating accel values we should reach zero
    // velocity at t=3.5 with the given data
    EXPECT_NEAR(x(3), 1.0, 1e-5);
    EXPECT_NEAR(x(6), 0.0, 1e-5);
}

TEST(InertialIntegratorTest,
     testRK4_inertialStateSimpleAccelAndSlowdownRotatedFrame)
{
    InertialIntegrator integrator;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);

    // Set initial orientation with y down
    x0(0) = -0.5;
    x0(1) = 0.5;
    x0(2) = -0.5;
    x0(3) = 0.5;

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x, zero_bias, G_gravity);
      };

    // Some fake data...
    // simple accel in z direction
    Eigen::Vector3d a(0, -8.81, 0);
    Eigen::Vector3d w(0, 0, 0);

    integrator.addData(0, a, w);
    integrator.addData(0.5, a, w);
    integrator.addData(1.0, a, w);
    integrator.addData(1.5, a, w);

    // slowdown!
    Eigen::Vector3d a2(0, -10.81, 0);
    integrator.addData(2.0, a2, w);
    integrator.addData(2.5, a2, w);
    integrator.addData(3.0, a2, w);
    integrator.addData(3.5, a2, w);

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 3.5, x0);

    // Because we're linearly interpolating accel values we should reach zero
    // velocity at t=3.5 with the given data
    EXPECT_NEAR(x(3), 0.5, 1e-5);
    EXPECT_NEAR(x(4), 0.0, 1e-5);
    EXPECT_NEAR(x(5), 0.0, 1e-5);
    EXPECT_NEAR(x(6), 0.0, 1e-5);
}

TEST(InertialIntegratorTest, testRK4_inertialStateSimulatedCircleTrajectory)
{
    InertialIntegrator integrator;

    // Read in simulated data
    std::string base_path = ros::package::getPath("semantic_slam");
    std::string time_file(fmt::format("{}/test/data/times.dat", base_path));
    std::string accel_file(
      fmt::format("{}/test/data/accel_meas.dat", base_path));
    std::string gyro_file(fmt::format("{}/test/data/gyro_meas.dat", base_path));

    auto times = dlmread(time_file);
    auto accels = dlmread(accel_file);
    auto omegas = dlmread(gyro_file);

    // std::cout << "times = \n" << times << std::endl;
    // std::cout << "accels = \n" << accels << std::endl;

    // add to integrator...
    for (size_t i = 0; i < times.rows(); ++i) {
        integrator.addData(times(i), accels.row(i), omegas.row(i));
    }

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0(3) = 1.0; // set identity quaternion
    // initial position = [50 0 50]
    x0(7) = 50;
    x0(9) = 50;

    Eigen::VectorXd x =
      integrator.integrateInertial(0, 40, x0, zero_bias, G_gravity);

    std::cout << "X final =\n" << x << std::endl;

    // from known truth values
    EXPECT_NEAR(x(3), -0.771482, 1e-3);

    EXPECT_NEAR(x(4), -16.98, 1e-1);
    EXPECT_NEAR(x(5), -20.49, 1e-1);
    EXPECT_NEAR(x(6), -16.98, 1e-1);
    EXPECT_NEAR(x(7), -38.492, 1);
    EXPECT_NEAR(x(8), 31.91, 1);
    EXPECT_NEAR(x(9), -38.492, 1);
}

TEST(InertialIntegratorTest, testRK4_inertialCircleTrajectoryWithCovariance)
{
    InertialIntegrator integrator;

    // Read in simulated data
    std::string base_path = ros::package::getPath("semantic_slam");
    std::string time_file(fmt::format("{}/test/data/times.dat", base_path));
    std::string accel_file(
      fmt::format("{}/test/data/accel_meas.dat", base_path));
    std::string gyro_file(fmt::format("{}/test/data/gyro_meas.dat", base_path));

    auto times = dlmread(time_file);
    auto accels = dlmread(accel_file);
    auto omegas = dlmread(gyro_file);

    // std::cout << "times = \n" << times << std::endl;
    // std::cout << "accels = \n" << accels << std::endl;

    // add to integrator...
    for (size_t i = 0; i < times.rows(); ++i) {
        integrator.addData(times(i), accels.row(i), omegas.row(i));
    }

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    x0(3) = 1.0; // set identity quaternion
    // initial position = [50 0 50]
    x0(7) = 50;
    x0(9) = 50;

    // These values are approx. the values of the IMU in the VI sensor
    integrator.setAdditiveMeasurementNoise({ 1e-4, 1e-4, 1e-4 },
                                           { 1.7e-3, 1.7e-3, 1.7e-3 });
    integrator.setBiasRandomWalkNoise({ 5e-5, 5e-5, 5e-5 },
                                      { 1e-3, 1e-3, 1e-3 });

    auto xP = integrator.integrateInertialWithCovariance(
      0, 40, x0, zero_bias, G_gravity);

    std::cout << "X final =\n" << xP[0] << std::endl;

    std::cout << "Covariance of q = \n" << xP[1].block<3, 3>(0, 0) << std::endl;
    std::cout << "Covariance of p = \n" << xP[1].block<3, 3>(6, 6) << std::endl;

    // from known truth values
    Eigen::VectorXd x = xP[0];
    EXPECT_NEAR(x(3), 0.771482, 1e-3);

    EXPECT_NEAR(x(4), -16.98, 1e-1);
    EXPECT_NEAR(x(5), -20.49, 1e-1);
    EXPECT_NEAR(x(6), -16.98, 1e-1);
    EXPECT_NEAR(x(7), -38.492, 1);
    EXPECT_NEAR(x(8), 31.91, 1);
    EXPECT_NEAR(x(9), -38.492, 1);

    // TODO compute truth values for P...
}

TEST(InertialIntegratorTest, testRK4_inertialJacobians)
{
    InertialIntegrator integrator;

    // Read in simulated data
    std::string base_path = ros::package::getPath("semantic_slam");
    std::string time_file(fmt::format("{}/test/data/times.dat", base_path));
    std::string accel_file(
      fmt::format("{}/test/data/accel_meas.dat", base_path));
    std::string gyro_file(fmt::format("{}/test/data/gyro_meas.dat", base_path));

    auto times = dlmread(time_file);
    auto accels = dlmread(accel_file);
    auto omegas = dlmread(gyro_file);

    // std::cout << "times = \n" << times << std::endl;
    // std::cout << "accels = \n" << accels << std::endl;

    // add to integrator...
    for (size_t i = 0; i < times.rows(); ++i) {
        integrator.addData(times(i), accels.row(i), omegas.row(i));
    }

    // These values are approx. the values of the IMU in the VI sensor
    integrator.setAdditiveMeasurementNoise({ 1e-4, 1e-4, 1e-4 },
                                           { 1.7e-3, 1.7e-3, 1.7e-3 });
    integrator.setBiasRandomWalkNoise({ 5e-5, 5e-5, 5e-5 },
                                      { 1e-3, 1e-3, 1e-3 });

    double t1 = 0;
    double t2 = 40;

    auto xJ = integrator.preintegrateInertialWithJacobianAndCovariance(
      t1, t2, zero_bias);

    std::cout << "X preint final =\n" << xJ[0].transpose() << std::endl;

    std::cout << "cov = \n" << xJ[2] << std::endl;

    // Compute actual X estimate from this preintegration
    Eigen::VectorXd xhat(10);
    xhat.head<4>() = xJ[0].topRows<4>();
    xhat.segment<3>(4) = xJ[0].middleRows<3>(4) + G_gravity * (t2 - t1);
    xhat.tail<3>() = xJ[0].bottomRows<3>() +
                     0.5 * G_gravity * (t2 - t1) * (t2 - t1) +
                     Eigen::Vector3d(50, 0, 50);

    std::cout << "X estimate final = \n" << xhat.transpose() << std::endl;

    // std::cout << "Jbias = \n" << xJ[1] << std::endl;

    // Compare the result of a small perturbation when multiplying the Jacobian
    // and when just re-doing the integration...
    Eigen::Vector3d delta_ba(0.01, 0.01, -0.01);
    Eigen::Vector3d delta_bg(0, 0, 0.004);
    Eigen::VectorXd delta_b(6);
    delta_b << delta_bg, delta_ba;

    auto x2 = integrator.preintegrateInertial(t1, t2, zero_bias + delta_b);

    std::cout << "X2 preint:\n" << x2.transpose() << std::endl;

    std::cout << "x + J*db:\n"
              << (xJ[0] + xJ[1] * delta_b).transpose() << std::endl;
}

namespace sym = symbol_shorthand;

// TEST(InertialIntegratorTest, testInertialFactor_OneFactorGraph)
// {
//     // InertialIntegrator integrator;

//     // Read in simulated data
//     std::string base_path = ros::package::getPath("semantic_slam");
//     std::string time_file(fmt::format("{}/test/data/times.dat", base_path));
//     std::string accel_file(
//       fmt::format("{}/test/data/accel_meas.dat", base_path));
//     std::string gyro_file(fmt::format("{}/test/data/gyro_meas.dat",
//     base_path));

//     auto times = dlmread(time_file);
//     auto accels = dlmread(accel_file);
//     auto omegas = dlmread(gyro_file);

//     FactorGraph graph;

//     SE3NodePtr origin_x =
//       util::allocate_aligned<SE3Node>(sym::X(0), ros::Time(0.0));
//     Vector3dNodePtr origin_v =
//       util::allocate_aligned<VectorNode<3>>(sym::V(0), ros::Time(0.0));
//     VectorNode<6>::Ptr origin_b =
//       util::allocate_aligned<VectorNode<6>>(sym::B(0), ros::Time(0.0));

//     origin_x->pose().rotation().coeffs() << 0, 0, 0, 1;
//     origin_x->pose().translation() << 50, 0, 50;

//     graph.addNodes({ origin_x, origin_v, origin_b });
//     graph.setNodesConstant({ origin_x, origin_v, origin_b });

//     // Have just two keyframes tmax seconds apart
//     double tmax = 4.0;
//     boost::shared_ptr<InertialIntegrator> integrator =
//       util::allocate_aligned<InertialIntegrator>();

//     // These values are approx. the values of the IMU in the VI sensor
//     integrator->setAdditiveMeasurementNoise(2e-4, 2e-3);
//     integrator->setBiasRandomWalkNoise(6e-5, 1e-3);

//     size_t idx = 0;
//     double t = 0;

//     for (; idx < times.rows(); ++idx) {
//         t = times(idx);

//         integrator->addData(t, accels.row(idx), omegas.row(idx));

//         if (t >= tmax)
//             break;
//     }

//     // Perform an integration and compute an initial estimate for the
//     // new frame's pose
//     Eigen::VectorXd last_qvp(10);
//     last_qvp.head<4>() = origin_x->pose().rotation().coeffs();
//     last_qvp.segment<3>(4) = origin_v->vector();
//     last_qvp.segment<3>(7) = origin_x->pose().translation();

//     Eigen::VectorXd xhat =
//       integrator->integrateInertial(0, t, last_qvp, origin_b->vector());

//     // Create the new graph nodes
//     ros::Time rost(t);
//     SE3NodePtr x = util::allocate_aligned<SE3Node>(sym::X(1), rost);
//     Vector3dNodePtr v = util::allocate_aligned<Vector3dNode>(sym::V(1),
//     rost); VectorNode<6>::Ptr b =
//       util::allocate_aligned<VectorNode<6>>(sym::B(1), rost);

//     // Compute estimates of the propagated pose
//     Pose3 G_T_new(Eigen::Quaterniond(xhat.head<4>()), xhat.tail<3>());
//     G_T_new.rotation().normalize();
//     if (G_T_new.rotation().w() < 0) {
//         G_T_new.rotation().coeffs() *= -1;
//     }

//     x->pose() = G_T_new;
//     v->vector() = xhat.segment<3>(4);
//     b->vector() = origin_b->vector();

//     x->pose().translation() += Eigen::Vector3d(.1, .1, .1);
//     x->pose().rotation().coeffs() += Eigen::Vector4d(0.01, 0.1, -0.01, 0);
//     x->pose().rotation().normalize();
//     v->vector() += Eigen::Vector3d(1, -0.1, 0.5);

//     // std::cout << "Pose at t = " << t << " is \n"
//     //           << x->pose() << "v: " << v->vector().transpose()
//     //           << "\nb: " << b->vector().transpose() << "\n"
//     //           << std::endl;

//     graph.addNodes({ x, v, b });

//     // Create the actual factor and add it
//     auto factor = util::allocate_aligned<CeresImuFactor>(
//       origin_x, origin_v, origin_b, x, v, b, integrator, 0, t);
//     graph.addFactor(factor);

//     // test test...
//     auto cf = factor->cf();
//     double residual[15];
//     double* parameters[] = {
//         origin_x->pose().data(),   origin_v->vector().data(),
//         origin_b->vector().data(), x->pose().data(),
//         v->vector().data(),        b->vector().data()
//     };
//     cf->Evaluate(parameters, residual, NULL);

//     std::cout << "Residual on adding = [";
//     for (int i = 0; i < 15; ++i) {
//         std::cout << residual[i] << " ";
//     }
//     std::cout << "];\n";

//     std::cout << "X origin:\n"
//               << origin_x->pose() << "v: " << origin_v->vector().transpose()
//               << "\nb: " << origin_b->vector().transpose() << "\n"
//               << std::endl
//               << std::endl;

//     std::cout << "X initial:\n"
//               << x->pose() << "v: " << v->vector().transpose()
//               << "\nb: " << b->vector().transpose() << "\n"
//               << std::endl
//               << std::endl;

//     std::cout << "X true:\n"
//               << G_T_new << "v: " << xhat.segment<3>(4).transpose()
//               << "\nb: " << origin_b->vector().transpose() << "\n"
//               << std::endl
//               << std::endl;

//     graph.solver_options().max_num_iterations = 100;
//     graph.solve(true);

//     std::cout << "After optimization:\n"
//               << x->pose() << "v: " << v->vector().transpose()
//               << "\nb: " << b->vector().transpose() << "\n"
//               << std::endl
//               << std::endl;

//     // Eigen::VectorXd x = integrator.integrateInertial(0, 40, x0,
//     zero_bias);

//     // std::cout << "X final =\n" << x << std::endl;

//     // // from known truth values
//     // EXPECT_NEAR(x(3), -0.771482, 1e-3);

//     // EXPECT_NEAR(x(4), -16.98, 1e-1);
//     // EXPECT_NEAR(x(5), -20.49, 1e-1);
//     // EXPECT_NEAR(x(6), -16.98, 1e-1);
//     // EXPECT_NEAR(x(7), -38.492, 1);
//     // EXPECT_NEAR(x(8), 31.91, 1);
//     // EXPECT_NEAR(x(9), -38.492, 1);
// }

TEST(InertialIntegratorTest, testInertialFactor_Construct)
{
    // InertialIntegrator integrator;

    // Read in simulated data
    std::string base_path = ros::package::getPath("semantic_slam");
    std::string time_file(fmt::format("{}/test/data/times.dat", base_path));
    std::string accel_file(
      fmt::format("{}/test/data/accel_meas.dat", base_path));
    std::string gyro_file(fmt::format("{}/test/data/gyro_meas.dat", base_path));

    auto times = dlmread(time_file);
    auto accels = dlmread(accel_file);
    auto omegas = dlmread(gyro_file);

    // add to integrator...
    // for (size_t i = 0; i < times.rows(); ++i) {
    //     integrator.addData(times(i), accels.row(i), omegas.row(i));
    // }

    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(10);
    // x0(3) = 1.0; // set identity quaternion
    // // initial position = [50 0 50]
    // x0(7) = 50;
    // x0(9) = 50;

    FactorGraph graph;

    SE3NodePtr origin_x =
      util::allocate_aligned<SE3Node>(sym::X(0), ros::Time(0.0));
    Vector3dNodePtr origin_v =
      util::allocate_aligned<VectorNode<3>>(sym::V(0), ros::Time(0.0));
    VectorNode<6>::Ptr origin_b =
      util::allocate_aligned<VectorNode<6>>(sym::B(0), ros::Time(0.0));

    Vector3dNodePtr gravity_node =
      util::allocate_aligned<Vector3dNode>(sym::R(0));
    gravity_node->vector() = G_gravity;

    origin_x->pose().rotation().coeffs() << 0, 0, 0, 1;
    origin_x->pose().translation() << 50, 0, 50;

    graph.addNodes({ origin_x, origin_v, origin_b, gravity_node });
    graph.setNodesConstant({ origin_x, origin_v, origin_b, gravity_node });

    // Say a keyframe every some seconds...
    double key_period = 0.1;
    double tmax = 8.0;
    double last_key_time = 0.0;
    SE3NodePtr last_x = origin_x;
    Vector3dNodePtr last_v = origin_v;
    VectorNode<6>::Ptr last_b = origin_b;
    int key_index = 1;
    boost::shared_ptr<InertialIntegrator> integrator =
      util::allocate_aligned<InertialIntegrator>();

    // These values are approx. the values of the IMU in the VI sensor
    integrator->setAdditiveMeasurementNoise({ 1e-4, 1e-4, 1e-4 },
                                            { 1.7e-3, 1.7e-3, 1.7e-3 });
    integrator->setBiasRandomWalkNoise({ 5e-5, 5e-5, 5e-5 },
                                       { 1e-3, 1e-3, 1e-3 });

    for (size_t i = 0; i < times.rows(); ++i) {
        double t = times(i);

        if (t > tmax)
            break;

        integrator->addData(t, accels.row(i), omegas.row(i));

        if (t >= last_key_time + key_period) {
            // Add a new keyframe
            // Perform an integration and compute an initial estimate for the
            // new frame's pose
            Eigen::VectorXd last_qvp(10);
            last_qvp.head<4>() = last_x->pose().rotation().coeffs();
            last_qvp.segment<3>(4) = last_v->vector();
            last_qvp.segment<3>(7) = last_x->pose().translation();

            Eigen::VectorXd xhat =
              integrator->integrateInertial(last_key_time,
                                            t,
                                            last_qvp,
                                            last_b->vector(),
                                            gravity_node->vector());

            // Create the new graph nodes
            ros::Time rost(t);
            SE3NodePtr x =
              util::allocate_aligned<SE3Node>(sym::X(key_index), rost);
            Vector3dNodePtr v =
              util::allocate_aligned<Vector3dNode>(sym::V(key_index), rost);
            VectorNode<6>::Ptr b =
              util::allocate_aligned<VectorNode<6>>(sym::B(key_index), rost);

            // Compute estimates of the propagated pose
            Pose3 G_T_new(Eigen::Quaterniond(xhat.head<4>()), xhat.tail<3>());
            if (G_T_new.rotation().w() < 0) {
                G_T_new.rotation().coeffs() *= -1;
            }

            x->pose() = G_T_new;
            v->vector() = xhat.segment<3>(4);
            b->vector() = last_b->vector();

            x->pose().translation() += Eigen::Vector3d(.1, .1, .1);
            x->pose().rotation().coeffs() +=
              Eigen::Vector4d(0.001, 0, -0.001, 0);
            x->pose().rotation().normalize();
            // v->vector() += Eigen::Vector3d(0.01, -0.1, 0.5);

            // std::cout << "Pose at t = " << t << " is \n"
            //           << x->pose() << "v: " << v->vector().transpose()
            //           << "\nb: " << b->vector().transpose() << "\n"
            //           << std::endl;

            graph.addNodes({ x, v, b });

            // Create the actual factor and add it
            auto factor = util::allocate_aligned<CeresImuFactor>(last_x,
                                                                 last_v,
                                                                 last_b,
                                                                 x,
                                                                 v,
                                                                 b,
                                                                 gravity_node,
                                                                 integrator,
                                                                 last_key_time,
                                                                 t);
            graph.addFactor(factor);

            // test test...
            // auto cf = factor->cf();
            // double residual[15];
            // double* parameters[] = {
            //     last_x->pose().data(),   last_v->vector().data(),
            //     last_b->vector().data(), x->pose().data(),
            //     v->vector().data(),      b->vector().data()
            // };
            // cf->Evaluate(parameters, residual, NULL);

            // std::cout << "Residual on adding = [";
            // for (int i = 0; i < 15; ++i) {
            //     std::cout << residual[i] << " ";
            // }
            // std::cout << "];\n";

            // Update variables for next keyframe
            last_x = x;
            last_v = v;
            last_b = b;
            key_index++;
            last_key_time = t;

            // integrator = util::allocate_aligned<InertialIntegrator>();
            // integrator->setAdditiveMeasurementNoise(1e-4, 1.7e-3);
            // integrator->setBiasRandomWalkNoise(5e-5, 1e-3);
            // integrator->addData(t, accels.row(i), omegas.row(i));
        }
    }

    std::cout << "X initial:\n"
              << last_x->pose() << "v: " << last_v->vector().transpose()
              << "\nb: " << last_b->vector().transpose()
              << "\ng: " << gravity_node->vector().transpose() << "\n"
              << std::endl
              << std::endl;

    // Compute truth values
    Eigen::VectorXd last_qvp(10);
    last_qvp.head<4>() = origin_x->pose().rotation().coeffs();
    last_qvp.segment<3>(4) = origin_v->vector();
    last_qvp.segment<3>(7) = origin_x->pose().translation();
    Eigen::VectorXd xtrue = integrator->integrateInertial(
      0, last_key_time, last_qvp, origin_b->vector(), gravity_node->vector());

    std::cout << "X true:\n"
              << Pose3(Eigen::Quaterniond(xtrue.head<4>()), xtrue.tail<3>())
              << "v: " << xtrue.segment<3>(4).transpose()
              << "\nb: " << origin_b->vector().transpose() << "\n"
              << std::endl
              << std::endl;

    graph.solver_options().max_num_iterations = 5000;

    graph.solver_options().trust_region_strategy_type = ceres::DOGLEG;
    graph.solver_options().dogleg_type = ceres::SUBSPACE_DOGLEG;

    // graph.solver_options().linear_solver_type = ceres::CGNR;
    // graph.solver_options().nonlinear_conjugate_gradient_type =
    //   ceres::POLAK_RIBIERE;

    // graph.solver_options().initial_trust_region_radius = 1e16;
    // graph.solver_options().use_inner_iterations = true;

    // graph.solver_options().use_nonmonotonic_steps = true;

    graph.solve(true);

    std::cout << "After optimization:\n"
              << last_x->pose() << "v: " << last_v->vector().transpose()
              << "\nb: " << last_b->vector().transpose()
              << "\ng: " << gravity_node->vector().transpose() << "\n"
              << std::endl
              << std::endl;

    // Eigen::VectorXd x = integrator.integrateInertial(0, 40, x0,
    // zero_bias);

    // std::cout << "X final =\n" << x << std::endl;

    // // from known truth values
    // EXPECT_NEAR(x(3), -0.771482, 1e-3);

    // EXPECT_NEAR(x(4), -16.98, 1e-1);
    // EXPECT_NEAR(x(5), -20.49, 1e-1);
    // EXPECT_NEAR(x(6), -16.98, 1e-1);
    // EXPECT_NEAR(x(7), -38.492, 1);
    // EXPECT_NEAR(x(8), 31.91, 1);
    // EXPECT_NEAR(x(9), -38.492, 1);
}

// Run all the tests that were declared with TEST()
int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}