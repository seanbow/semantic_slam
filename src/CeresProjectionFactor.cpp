#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_projection.h"

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/ProjectionFactor.h>

CeresProjectionFactor::CeresProjectionFactor(
  SE3NodePtr camera_node,
  Vector3dNodePtr landmark_node,
  const Eigen::Vector2d& image_coords,
  const Eigen::Matrix2d& msmt_covariance,
  boost::shared_ptr<CameraCalibration> calibration,
  const Pose3& body_T_sensor,
  bool use_huber,
  int tag)
  : CeresFactor(FactorType::PROJECTION, tag)
  , image_coords_(image_coords)
  , covariance_(msmt_covariance)
  , calibration_(calibration)
  , body_T_sensor_(body_T_sensor)
  , robust_loss_(use_huber)
{
    cf_ = ProjectionCostTerm::Create(image_coords,
                                     msmt_covariance,
                                     body_T_sensor.rotation(),
                                     body_T_sensor.translation(),
                                     calibration);

    nodes_.push_back(camera_node);
    nodes_.push_back(landmark_node);

    createGtsamFactor();
}

CeresFactor::Ptr
CeresProjectionFactor::clone() const
{
    return util::allocate_aligned<CeresProjectionFactor>(nullptr,
                                                         nullptr,
                                                         image_coords_,
                                                         covariance_,
                                                         calibration_,
                                                         body_T_sensor_,
                                                         robust_loss_,
                                                         tag_);
}

CeresProjectionFactor::~CeresProjectionFactor()
{
    delete cf_;
}

void
CeresProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    ceres::LossFunction* lf;
    if (robust_loss_) {
        lf = new ceres::HuberLoss(1.5);
    } else {
        lf = NULL;
    }

    active_ = true;

    ceres::ResidualBlockId residual_id = problem->AddResidualBlock(
      cf_, lf, camera_node()->pose().data(), landmark_node()->vector().data());

    residual_ids_[problem.get()] = residual_id;
}

void
CeresProjectionFactor::createGtsamFactor() const
{
    if (camera_node() && landmark_node()) {

        auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);

        // our calibration == nullptr corresponds to an already calibrated
        // camera, i.e. cx = cy = 0 and fx = fy = 1, which is what the default
        // gtsam calibration constructor provides
        auto gtsam_calib =
          calibration_ ? util::allocate_aligned<gtsam::Cal3DS2>(*calibration_)
                       : util::allocate_aligned<gtsam::Cal3DS2>();

        gtsam_factor_ = util::allocate_aligned<
          gtsam::GenericProjectionFactor<gtsam::Pose3,
                                         gtsam::Point3,
                                         gtsam::Cal3DS2>>(
          image_coords_,
          gtsam_noise,
          camera_node()->key(),
          landmark_node()->key(),
          gtsam_calib,
          gtsam::Pose3(body_T_sensor_));
    }
}

void
CeresProjectionFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    if (!gtsam_factor_)
        createGtsamFactor();

    graph->push_back(gtsam_factor_);
}