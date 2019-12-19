#pragma once

#include "semantic_slam/Camera.h"
#include "semantic_slam/Common.h"

enum class TriangulationStatus
{
    SUCCESS,
    BEHIND_CAMERA,
    DEGENERATE,
    OUTLIER,
    FAILURE
};

struct TriangulationResult
{
    Eigen::Vector3d point;
    TriangulationStatus status;
    double max_reprojection_error;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class CameraSet
{
  public:
    CameraSet() {}

    void addCamera(const Camera& camera) { cameras_.push_back(camera); }

    aligned_vector<Camera>& cameras() { return cameras_; }
    const aligned_vector<Camera>& cameras() const { return cameras_; }

    TriangulationResult triangulateMeasurements(
      const aligned_vector<Eigen::Vector2d>& pixel_msmts,
      boost::optional<double&> condition_number = boost::none);

    TriangulationResult triangulateMeasurementsApproximate(
      const aligned_vector<Eigen::Vector2d>& pixel_msmts,
      int n_cameras = 50,
      boost::optional<double&> condition_number = boost::none);

    TriangulationResult triangulateIterative(
      const aligned_vector<Eigen::Vector2d>& pixel_msmts);

  private:
    aligned_vector<Camera> cameras_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};