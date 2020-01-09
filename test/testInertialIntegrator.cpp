

#include "semantic_slam/inertial/InertialIntegrator.h"

#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include <ros/package.h>
#include <rosfmt/rosfmt.h>

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

TEST(InertialIntegratorTest, testRK4_basic)
{
    InertialIntegrator integrator;

    std::function<double(double, const double&)> f =
      [](double t, const double& x) -> double { return x; };

    // integration of xdot = f(t,x) = x from t = 0 to t = T should be e^2...

    double e2 = integrator.integrateRK4(f, 0, 2, 1.0);

    EXPECT_NEAR(e2, std::exp(2), 1e-4);
}

TEST(InertialIntegratorTest, testRK4_vector)
{
    InertialIntegrator integrator;

    Eigen::Matrix2d A;
    A << 1, 2, 3, 4;

    std::function<Eigen::Vector2d(double, const Eigen::Vector2d&)> f =
      [&](double t, Eigen::Vector2d x) { return A * x; };

    Eigen::Vector2d x0;
    x0 << 1, 1;

    Eigen::Vector2d x = integrator.integrateRK4(f, 0, 1, x0, 0.02);

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
          return integrator.statedot(t, x);
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
          return integrator.statedot(t, x);
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
          return integrator.statedot(t, x);
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
          return integrator.statedot(t, x);
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
          return integrator.statedot(t, x);
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

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)> statedot =
      [&](double t, const Eigen::VectorXd& x) {
          return integrator.statedot(t, x);
      };

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

    Eigen::VectorXd x = integrator.integrateRK4(statedot, 0, 40, x0, 0.01);

    std::cout << "X final =\n" << x << std::endl;

    // // Because we're linearly interpolating accel values we should reach zero
    // // velocity at t=3.5 with the given data
    // EXPECT_NEAR(x(3), 1.0, 1e-5);
    // EXPECT_NEAR(x(6), 0.0, 1e-5);
}

// Run all the tests that were declared with TEST()
int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}