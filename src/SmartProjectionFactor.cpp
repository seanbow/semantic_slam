#include "semantic_slam/SmartProjectionFactor.h"

#include <gtsam/slam/SmartProjectionPoseFactor.h>

SmartProjectionFactor::SmartProjectionFactor(const Pose3& body_T_sensor,
                                            boost::shared_ptr<CameraCalibration> calibration,
                                            double reprojection_error_threshold,
                                            int tag)
    : CeresFactor(FactorType::SMART_PROJECTION, tag),
      I_T_C_(body_T_sensor),
      calibration_(calibration),
      reprojection_error_threshold_(reprojection_error_threshold),
      in_graph_(false),
      active_(false),
      problem_(nullptr),
      triangulation_good_(false)
{
    // Parameter block ordering:
    // camera poses (q1 p1), (q2 p2), ...
    // [q1 p1 q2 p2 ... qn pn]

    // TODO get the real value
    auto gtsam_noise = gtsam::noiseModel::Isotropic::Sigma(2, 4);

    gtsam::SmartProjectionParams projection_params;
    projection_params.degeneracyMode = gtsam::DegeneracyMode::ZERO_ON_DEGENERACY;
    projection_params.linearizationMode = gtsam::LinearizationMode::HESSIAN;
    projection_params.setLandmarkDistanceThreshold(1e6);
    projection_params.setRankTolerance(1e-2);
    projection_params.triangulation.dynamicOutlierRejectionThreshold = reprojection_error_threshold;

    gtsam_factor_ = util::allocate_aligned<GtsamFactorType>(gtsam_noise,
                                                            util::allocate_aligned<gtsam::Cal3DS2>(*calibration),
                                                            gtsam::Pose3(body_T_sensor),
                                                            projection_params);
                                                            
}

size_t SmartProjectionFactor::nMeasurements() const
{
    return msmts_.size();
}

bool SmartProjectionFactor::decideIfTriangulate(const aligned_vector<Pose3>& body_poses) const
{
    // Triangulate if:
    //  (1) the last triangulation failed,
    //  (2) camera poses are sufficiently different from last time,
    //  (3) or, the number of camera poses is different.
    if (!triangulation_good_ || body_poses.size() != triangulation_poses_.size()) {
        return true;
    }

    // Compare each camera pose
    bool equal = true;
    for (int i = 0; i < body_poses.size(); ++i) {
        Pose3 camera_pose = body_poses[i].compose(I_T_C_);

        if (!camera_pose.equals(triangulation_poses_[i], 1e-2)) {
            equal = false;
            break;
        }
    }

    return equal;
}

void SmartProjectionFactor::triangulate(const aligned_vector<Pose3>& body_poses) const
{
    triangulation_good_ = false;
    triangulation_poses_.clear();

    CameraSet cameras;
    for (int i = 0; i < msmts_.size(); ++i) {
        triangulation_poses_.push_back(body_poses[i].compose(I_T_C_));
        Camera camera(triangulation_poses_[i], calibration_);
        cameras.addCamera(camera);
    }

    // double cond;
    TriangulationResult triangulation = cameras.triangulateMeasurementsApproximate(msmts_, 25);

    if (triangulation.max_reprojection_error <= reprojection_error_threshold_ &&
            triangulation.status == TriangulationStatus::SUCCESS) {
        landmark_position_ = triangulation.point;
        triangulation_good_ = true;
    }

    // std::cout << "Triangulation = " << triangulation.point.transpose() << std::endl;
}

void SmartProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // TODO add huber loss
    in_graph_ = true;
    problem_ = problem;

    if (triangulation_good_) {
        residual_id_ = problem->AddResidualBlock(this, NULL, parameter_blocks_);
        active_ = true;
    }
}

void SmartProjectionFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->RemoveResidualBlock(residual_id_);

    active_ = false;
}


void SmartProjectionFactor::addMeasurement(SE3NodePtr body_pose_node,
                                        const Eigen::Vector2d& pixel_coords, 
                                        const Eigen::Matrix2d& msmt_covariance)
{
    body_poses_.push_back(body_pose_node);
    msmts_.push_back(pixel_coords);
    // covariances_.push_back(msmt_covariance);

    sqrt_informations_.push_back(Eigen::Matrix2d::Identity());
    Eigen::Matrix2d sqrtC = msmt_covariance.llt().matrixL();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_informations_.back());

    // Ceres can't handle block sizes changing if we've already been added to the Problem.
    // So we need to remove and re-add ourself
    if (in_graph_ && active_) {
        removeFromProblem(problem_);
    }

    mutable_parameter_block_sizes()->push_back(4); // q
    parameter_blocks_.push_back(body_pose_node->pose().rotation_data());
    mutable_parameter_block_sizes()->push_back(3); // p
    parameter_blocks_.push_back(body_pose_node->pose().translation_data());

    set_num_residuals(2 * nMeasurements() - 3);        
    
    aligned_vector<Pose3> body_poses;
    for (auto& node : body_poses_) {
        body_poses.push_back(node->pose());
    }

    triangulate(body_poses);

    if (in_graph_ && triangulation_good_) {
        addToProblem(problem_);
    }

    // gtsam support
    gtsam_factor_->add(pixel_coords, body_pose_node->key());
}

bool SmartProjectionFactor::Evaluate(double const* const* parameters, 
                                     double* residuals_ptr, 
                                     double** jacobians) const
{
    // Collect parameters
    aligned_vector<Pose3> body_poses;
    for (int i = 0; i < nMeasurements(); ++i) {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[2*i]);
        Eigen::Map<const Eigen::Vector3d> p(parameters[2*i + 1]);

        body_poses.push_back(Pose3(q,p));
    }

    Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, num_residuals());

    if (decideIfTriangulate(body_poses)) {
        triangulate(body_poses);
    }

    // Iterate through each measurement computing its residual & jacobian
    // If we need the Jacobians, compute them as we go...
    // Indices corresponding to this are 0 (pt), 2*i, and 1 + 2*i...
    using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Begin by collecting all the jacobians
    // We have to do this whether ceres is requesting the Jacobians or not because 
    // we need to project the residual into the point jacobian's null space
    Eigen::MatrixXd Hpoint = Eigen::MatrixXd::Zero(2 * nMeasurements(), 3);
    Eigen::MatrixXd Hpose = Eigen::MatrixXd::Zero(2 * nMeasurements(), 7 * nMeasurements());
    Eigen::VectorXd full_residual = Eigen::VectorXd::Zero(2 * nMeasurements());

    for (int i = 0; i < nMeasurements(); ++i) {
        Eigen::MatrixXd Hpose_compose;
        Pose3 G_T_C = body_poses[i].compose(I_T_C_, Hpose_compose, boost::none);

        Camera cam(G_T_C, calibration_);

        Eigen::MatrixXd Hpose_project, Hpoint_project;
        Eigen::Vector2d zhat;
        
        try {
            zhat = cam.project(landmark_position_, Hpose_project, Hpoint_project);

            Hpose.block<2,7>(2*i, 7*i) = -sqrt_informations_[i] * Hpose_project * Hpose_compose;
            Hpoint.block<2,3>(2*i, 0) = -sqrt_informations_[i] * Hpoint_project;

            full_residual.segment<2>(2*i) = sqrt_informations_[i] * (msmts_[i] - zhat);
        } catch (CheiralityException& e) {
            // ignore this measurement for now
            // jacobians, residual already zeroed
        }
    } 


    // Jacobian matrices are filled in
    // Compute the basis of Hpoint's left null space

    // Eigen::HouseholderQR<Eigen::MatrixXd> qr(Hpoint);
    // Eigen::MatrixXd Q = qr.householderQ();
    // Eigen::MatrixXd basis = Q.rightCols(num_residuals());

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(Hpoint.transpose());
    Eigen::MatrixXd V = cod.matrixZ().transpose();
    // Eigen::MatrixXd basis = V.block(0, cod.rank(), V.rows(), V.cols() - cod.rank());
    Eigen::MatrixXd basis = V.block(0, 3, V.rows(), num_residuals());
    basis.applyOnTheLeft(cod.colsPermutation());

    Hpose.applyOnTheLeft(basis.transpose());
    // full_residual.applyOnTheLeft(basis.transpose());

    // Fill in Ceres jacobian data pointers
    if (jacobians) {
        for (int i = 0; i < nMeasurements(); ++i) {
            if (jacobians[2*i]) {
                Eigen::Map<JacobianType> Dr_dq(jacobians[2*i], num_residuals(), 4);
                Dr_dq = Hpose.block(0, 7*i, num_residuals(), 4);
            }

            if (jacobians[2*i + 1]) {
                Eigen::Map<JacobianType> Dr_dp(jacobians[2*i + 1], num_residuals(), 3);
                Dr_dp = Hpose.block(0, 7*i + 4, num_residuals(), 3);
            }
        }
    }

    residuals = basis.transpose() * full_residual;

    return true;
}


boost::shared_ptr<gtsam::NonlinearFactor> 
SmartProjectionFactor::getGtsamFactor() const
{
    return gtsam_factor_;
}