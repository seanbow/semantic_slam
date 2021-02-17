#pragma once

#include <boost/optional.hpp>

#include <gtsam/geometry/Pose3.h>

class Pose3
{
  public:
    Pose3();
    Pose3(Eigen::Quaterniond q, Eigen::Vector3d p);

    Pose3(const gtsam::Pose3& pose);

    Pose3(const Pose3& other);
    Pose3& operator=(const Pose3& other);

    // Do we need these?
    Pose3(Pose3&& other);
    Pose3& operator=(Pose3&& other);

    Pose3(const Eigen::VectorXd& data_vec);

    static Pose3 Identity();

    const Eigen::Map<Eigen::Quaterniond>& rotation() const { return q_; }
    Eigen::Map<Eigen::Quaterniond>& rotation() { return q_; }

    const Eigen::Map<Eigen::Vector3d>& translation() const { return p_; }
    Eigen::Map<Eigen::Vector3d>& translation() { return p_; }

    double* data() { return data_vector_.data(); }
    double* rotation_data() { return data_vector_.data(); }
    double* translation_data() { return data_vector_.data() + 4; }

    const double* data() const { return data_vector_.data(); }
    const double* rotation_data() const { return data_vector_.data(); }
    const double* translation_data() const { return data_vector_.data() + 4; }

    Eigen::Vector3d transform_from(
      const Eigen::Vector3d& p,
      boost::optional<Eigen::MatrixXd&> Hpose = boost::none,
      boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    Eigen::Vector3d transform_to(
      const Eigen::Vector3d& p,
      boost::optional<Eigen::MatrixXd&> Hpose = boost::none,
      boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    Pose3 inverse(boost::optional<Eigen::MatrixXd&> H = boost::none) const;

    Pose3 compose(const Pose3& other,
                  boost::optional<Eigen::MatrixXd&> H1 = boost::none,
                  boost::optional<Eigen::MatrixXd&> H2 = boost::none) const;

    Pose3 operator*(const Pose3& other) const;
    Eigen::Vector3d operator*(const Eigen::Vector3d& other_pt) const;

    Pose3 between(const Pose3& other,
                  boost::optional<Eigen::MatrixXd&> Hpose1 = boost::none,
                  boost::optional<Eigen::MatrixXd&> Hpose2 = boost::none) const;

    bool equals(const Pose3& other, double tolerance) const;

    double x() const { return translation()[0]; }
    double y() const { return translation()[1]; }
    double z() const { return translation()[2]; }

    friend std::ostream& operator<<(std::ostream& os, const Pose3& p);

    operator gtsam::Pose3() const;

  private:
    // This sort of unfortunate data structure layout is needed for easier interfacing with Ceres
    Eigen::Matrix<double, 7, 1> data_vector_;
    Eigen::Map<Eigen::Quaterniond> q_;
    Eigen::Map<Eigen::Vector3d> p_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
