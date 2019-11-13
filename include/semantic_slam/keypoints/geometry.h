#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "semantic_slam/Common.h"
#include "semantic_slam/keypoints/geometry.h"

#include <fstream>
#include <limits>

#include <boost/shared_ptr.hpp>

namespace geometry
{
struct StructureResult
{
  // Eigen::Matrix<double, 3, Eigen::Dynamic> S;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  Eigen::VectorXd C;

  Eigen::VectorXd Z;
  Eigen::VectorXd Z_covariance;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ObjectModelBasis
{
  // mean shape
  Eigen::MatrixXd mu;

  // deformable basis components
  Eigen::MatrixXd pc;
};

using ObjectModelBasisPtr = boost::shared_ptr<ObjectModelBasis>;

ObjectModelBasis readModelFile(std::string file_name);

Eigen::MatrixXd centralize(const Eigen::MatrixXd& M);

Eigen::MatrixXd reshapeS_b2v(const Eigen::MatrixXd& S);

Eigen::MatrixXd composeShape(const Eigen::MatrixXd& B, const Eigen::VectorXd& C);

Eigen::VectorXd sample_stddev(const Eigen::MatrixXd& data);

StructureResult optimizeStructureFromProjection(const Eigen::MatrixXd& normalized_coords, const geometry::ObjectModelBasis& model,
                                                Eigen::VectorXd weights, bool compute_covariance = false);

StructureResult optimizeStructureFromProjectionCoordinateDescent(const Eigen::MatrixXd& normalized_coords,
                                                                 const geometry::ObjectModelBasis& model,
                                                                 const Eigen::VectorXd& weights);

// Returns +1 if val > 0, 0 if val == 0, -1 if val < 0.
template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

}  // namespace geometry
