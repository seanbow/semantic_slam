#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_projection.h"

CeresProjectionFactor::CeresProjectionFactor(SE3NodePtr camera_node,
                          Vector3dNodePtr landmark_node,
                          const Eigen::Vector2d& image_coords,
                          const Eigen::Matrix2d& msmt_covariance,
                          boost::shared_ptr<CameraCalibration> calibration,
                          const Pose3& body_T_sensor,
                          bool use_huber,
                          int tag)
    : CeresFactor(FactorType::PROJECTION, tag),
      camera_node_(camera_node),
      landmark_node_(landmark_node),
      robust_loss_(use_huber)
{
    cf_ = ProjectionCostTerm::Create(image_coords, 
                                     msmt_covariance, 
                                     body_T_sensor.rotation(),
                                     body_T_sensor.translation(),
                                     calibration);
}

CeresProjectionFactor::~CeresProjectionFactor()
{
    delete cf_;
}

void CeresProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    ceres::LossFunction* lf;
    if (robust_loss_) {
        lf = new ceres::HuberLoss(1.5);
    } else {
        lf = NULL;
    }

    residual_id_ = problem->AddResidualBlock(cf_, 
                                            lf, 
                                            camera_node_->pose().rotation_data(), 
                                            camera_node_->pose().translation_data(), 
                                            landmark_node_->vector().data());
}


void CeresProjectionFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->RemoveResidualBlock(residual_id_);
}
