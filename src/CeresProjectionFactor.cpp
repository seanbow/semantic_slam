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
  , camera_node_(camera_node)
  , landmark_node_(landmark_node)
  , robust_loss_(use_huber)
  , image_coords_(image_coords)
  , covariance_(msmt_covariance)
  , calibration_(calibration)
  , body_T_sensor_(body_T_sensor)
{
    cf_ = ProjectionCostTerm::Create(image_coords,
                                     msmt_covariance,
                                     body_T_sensor.rotation(),
                                     body_T_sensor.translation(),
                                     calibration);

    nodes_.push_back(camera_node);
    nodes_.push_back(landmark_node);

    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);
    auto gtsam_calib = util::allocate_aligned<gtsam::Cal3DS2>(*calibration_);

    gtsam_factor_ =
      util::allocate_aligned<gtsam::GenericProjectionFactor<gtsam::Pose3,
                                                            gtsam::Point3,
                                                            gtsam::Cal3DS2>>(
        image_coords,
        gtsam_noise,
        camera_node->key(),
        landmark_node->key(),
        gtsam_calib,
        gtsam::Pose3(body_T_sensor));
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

    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_,
                                lf,
                                camera_node_->pose().rotation_data(),
                                camera_node_->pose().translation_data(),
                                landmark_node_->vector().data());

    residual_ids_[problem.get()] = residual_id;
}

boost::shared_ptr<gtsam::NonlinearFactor>
CeresProjectionFactor::getGtsamFactor() const
{
    return gtsam_factor_;
}

void
CeresProjectionFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    graph->push_back(getGtsamFactor());
}