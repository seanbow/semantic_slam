
#include "semantic_slam/keypoints/EstimatedObject.h"
// #include "omnigraph/keypoints/StructureFactor.h"
#include "semantic_slam/Camera.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/geometry.h"

#include <unordered_set>

// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/inference/Key.h>
// #include <gtsam/inference/Symbol.h>
// #include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/nonlinear/NonlinearEquality.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/slam/PriorFactor.h>

#include <fmt/format.h>

namespace sym = symbol_shorthand;

EstimatedObject::EstimatedObject(
  boost::shared_ptr<FactorGraph> graph,
  boost::shared_ptr<FactorGraph> semantic_graph,
  const ObjectParams& params,
  geometry::ObjectModelBasis object_model,
  uint64_t object_id,
  uint64_t first_keypoint_id,
  const ObjectMeasurement& msmt,
  const Pose3& G_T_C,
  const Pose3& I_T_C,
  std::string platform,
  boost::shared_ptr<CameraCalibration> calibration,
  SemanticMapper* mapper)
  : graph_(graph)
  , semantic_graph_(semantic_graph)
  , id_(object_id)
  , first_kp_id_(first_keypoint_id)
  , obj_name_(msmt.obj_name)
  , in_graph_(false)
  , is_bad_(false)
  // , in_worldmodel_(false)
  , last_seen_(Symbol(msmt.observed_key).index())
  ,
  // last_visible_(msmt.pose_id),
  last_optimized_(Symbol(msmt.observed_key).index())
  , params_(params)
  , I_T_C_(I_T_C)
  , platform_(platform)
  , camera_calibration_(calibration)
  , model_(object_model)
  , modified_(false)
  , mapper_(mapper)
{
    // NOTE : because we need shared_from_this() to create the child keypoints,
    //  we cannot call the initialization method from within the constructor.
    // But also it doesn't make sense to have such an uninitialized object.
    // --> Make this constructor private and require the use of a create()
    // method. initializeFromMeasurement(msmt, G_T_C);

    // std::string
    // model_path("/home/sean/rcta/src/mapping/semslam/semslam/models/car_basis.dat");
    // model_ = geometry::readModelFile(model_path);

    m_ = model_.mu.cols();
    k_ = model_.pc.rows() / 3;

    last_visible_ = Symbol(msmt.observed_key).index();

    graph_pose_node_ = util::allocate_aligned<SE3Node>(sym::O(id_));

    if (k_ > 0) {
        graph_coefficient_node_ = util::allocate_aligned<VectorXdNode>(
          sym::C(id_),
          boost::none, // no associated time
          k_);
        graph_coefficient_node_->vector().setZero();
    }
}

EstimatedObject::Ptr
EstimatedObject::Create(boost::shared_ptr<FactorGraph> graph,
                        boost::shared_ptr<FactorGraph> semantic_graph,
                        const ObjectParams& params,
                        geometry::ObjectModelBasis object_model,
                        uint64_t object_id,
                        uint64_t first_keypoint_id,
                        const ObjectMeasurement& msmt,
                        const Pose3& G_T_C,
                        const Pose3& I_T_C,
                        std::string platform,
                        boost::shared_ptr<CameraCalibration> calibration,
                        SemanticMapper* mapper)
{
    EstimatedObject::Ptr pt(new EstimatedObject(graph,
                                                semantic_graph,
                                                params,
                                                object_model,
                                                object_id,
                                                first_keypoint_id,
                                                msmt,
                                                G_T_C,
                                                I_T_C,
                                                platform,
                                                calibration,
                                                mapper));
    pt->initializeFromMeasurement(msmt, G_T_C);
    return pt;
}

void
EstimatedObject::initializeFromMeasurement(const ObjectMeasurement& msmt,
                                           const Pose3& G_T_C)
{
    initializePose(msmt, G_T_C);
    initializeKeypoints(msmt);
    initializeStructure(msmt);
}

void
EstimatedObject::initializePose(const ObjectMeasurement& msmt,
                                const Pose3& G_T_C)
{
    Pose3 C_T_O = Pose3(msmt.q, msmt.t);
    // pose_ = G_T_C.compose(C_T_O);
    // graph_pose_node_->pose() = G_T_C.compose(C_T_O);
    pose_ = G_T_C.compose(C_T_O);

    // ROS_WARN_STREAM("Initialized object with pose:\n" << pose_);
}

void
EstimatedObject::initializeKeypoints(const ObjectMeasurement& msmt)
{
    if (in_graph_)
        throw std::runtime_error(
          "Cannot initialize an object already in the pose graph");

    if (keypoints_.empty()) {
        for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
            EstimatedObject::Ptr shared_this = shared_from_this();

            EstimatedKeypoint::Ptr kp(
              new EstimatedKeypoint(graph_,
                                    semantic_graph_,
                                    params_,
                                    first_kp_id_ + i,
                                    id_,
                                    msmt.keypoint_measurements[i].kp_class_id,
                                    I_T_C_,
                                    platform_,
                                    camera_calibration_,
                                    shared_this,
                                    mapper_));
            keypoints_.push_back(kp);

            kp->initializeFromMeasurement(msmt.keypoint_measurements[i]);
        }
    }
}

void
EstimatedObject::initializeStructure(const ObjectMeasurement& msmt)
{
    // Build list of keypoint nodes
    if (keypoints_.empty()) {
        throw std::runtime_error(
          "Tried to initialize object structure before keypoints");
    }

    std::vector<Vector3dNodePtr> keypoint_nodes;

    for (size_t i = 0; i < keypoints_.size(); ++i) {
        keypoint_nodes.push_back(keypoints_[i]->graph_node());
    }

    Key coeff_key = sym::C(id());

    // Just initializing -- we have no good keypoint estimates yet and hence no
    // good weights
    Eigen::VectorXd w = params_.structure_error_coefficient *
                        Eigen::VectorXd::Ones(keypoints_.size());

    structure_factor_ = util::allocate_aligned<CeresStructureFactor>(
      graph_pose_node_,
      keypoint_nodes,
      graph_coefficient_node_,
      model_,
      w,
      params_.structure_regularization_factor);
}

bool
EstimatedObject::inGraph() const
{
    return in_graph_;
}

int64_t
EstimatedObject::findKeypointByClass(uint64_t classid) const
{
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        if (keypoints_[i]->classid() == classid)
            return i;
    }
    return -1;
}

int64_t
EstimatedObject::findKeypointByKey(Key key) const
{
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        if (sym::L(keypoints_[i]->id()) == key) {
            return i;
        }
    }
    return -1;
}

// Pose3
// EstimatedObject::pose() const
// {
//   return graph_pose_node_->pose();
// }

void
EstimatedObject::optimizeStructure()
{
    size_t m = model_.mu.cols();
    size_t k = model_.pc.rows() / 3;

    Eigen::MatrixXd mu = geometry::centralize(model_.mu);
    Eigen::MatrixXd pc = geometry::centralize(model_.pc);

    double lambda = params_.structure_regularization_factor;

    Eigen::VectorXd weights = Eigen::VectorXd::Ones(m);
    Pose3 body_T_camera(I_T_C_.rotation(), I_T_C_.translation());

    // Pose3 initial_object_pose = graph_pose_node_->pose();
    Pose3 initial_object_pose = pose_;

    std::lock_guard<std::mutex> lock(problem_mutex_);

    structure_problem_ = util::allocate_aligned<StructureOptimizationProblem>(
      model_, camera_calibration_, body_T_camera, weights, params_);

    structure_problem_->initializePose(initial_object_pose);

    for (auto& obj_msmt : measurements_) {
        auto keyframe = mapper_->getKeyframeByKey(obj_msmt.observed_key);
        // auto cam_node = graph_->getNode<SE3Node>(obj_msmt.observed_key);
        if (!keyframe) {
            ROS_ERROR_STREAM("Unable to find graph node for camera pose "
                             << DefaultKeyFormatter(obj_msmt.observed_key));
        }

        // Pose3 cam_pose = keyframe->pose();

        // Eigen::Matrix<double, 6, 1> prior_noise_vec;
        // prior_noise_vec << 1e-3 * Eigen::Vector3d::Ones(),
        //                    1e-2 * Eigen::Vector3d::Ones();

        structure_problem_->addCamera(keyframe, true);
    }

    for (auto& kp : keypoints_) {
        structure_problem_->initializeKeypointPosition(kp->classid(),
                                                       kp->position());

        int n_meas = 0;

        for (auto& kp_msmt : kp->measurements()) {
            if (kp_msmt.observed) {
                structure_problem_->addKeypointMeasurement(kp_msmt);
                n_meas++;
            }
        }

        // std::cout << "Keypoint " << kp->id() << " has " << n_meas << "
        // measurements." << std::endl;
    }

    structure_problem_->solve();

    pose_ = structure_problem_->getObjectPose();

    // graph_pose_node_->pose() = pose_;

    if (k > 0)
        basis_coefficients_ = structure_problem_->getBasisCoefficients();

    for (auto& kp : keypoints_) {
        if (!kp->bad())
            kp->position() = *structure_problem_->getKeypoint(kp->classid());
    }

    // std::cout << "Optimization result -->\n" << pose_ << std::endl;
    // std::cout << "c = " << basis_coefficients_.transpose() << std::endl;
}

// double
// EstimatedObject::getStructureError() const
// {
//   return structure_graph_.error(structure_optimization_values_);
// }

Eigen::MatrixXd
EstimatedObject::getPlx(Key o_key, Key x_key) const
{
    // if (in_graph_) {
    //   return mapper_->getPlx(o_key, x_key);
    // }

    if (!structure_problem_) {
        throw std::logic_error("Tried to extract covariance information before "
                               "structure optimization.");
    }

    Symbol x_symbol(x_key);

    std::lock_guard<std::mutex> lock(problem_mutex_);
    Eigen::MatrixXd Plx = structure_problem_->getPlx(x_symbol.index());

    return Plx;
}

double
EstimatedObject::computeMahalanobisDistance(const ObjectMeasurement& msmt) const
{
    // if (msmt.class_id != classid_) return std::numeric_limits<double>::max();
    // ROS_WARN_STREAM("Computing mahal for object " << id());
    if (msmt.obj_name != obj_name_) {
        // ROS_WARN_STREAM("Object id " << id() << " name " << obj_name_
        //                              << " != measurement class " <<
        //                              msmt.obj_name);
        return std::numeric_limits<double>::max();
    }

    auto keyframe = mapper_->getKeyframeByKey(msmt.observed_key);
    Pose3 G_T_I = keyframe->pose();
    Eigen::MatrixXd Hpose_compose;
    Camera camera(G_T_I.compose(I_T_C_, Hpose_compose), camera_calibration_);

    Eigen::MatrixXd Plx = mapper_->getPlx(sym::O(id()), msmt.observed_key);

    // assemble the keypoint measurement vector & Jacobians
    Eigen::VectorXd msmt_vec = Eigen::VectorXd::Zero(2 * keypoints_.size());
    Eigen::VectorXd residuals = Eigen::VectorXd::Zero(2 * keypoints_.size());
    Eigen::MatrixXd H =
      Eigen::MatrixXd::Zero(2 * keypoints_.size(), Plx.rows());
    Eigen::VectorXd R_sqrt_vec = Eigen::VectorXd::Zero(2 * keypoints_.size());

    // Hpose will be in the *ambient* (4-dimensional) quaternion space.
    // Want it in the *tangent* (3-dimensional) space.
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> Hquat_space;
    QuaternionLocalParameterization().ComputeJacobian(G_T_I.rotation_data(),
                                                      Hquat_space.data());

    // Index of x into Plx...
    size_t x_index = 3 * keypoints_.size();

    size_t n_observed = 0;
    for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
        if (!msmt.keypoint_measurements[i].observed) {
            // We can just leave the residual vector zero & pick a
            // lower-dimensional chi-2 distribution
            continue;
        }

        int kp_index =
          findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id);
        if (kp_index < 0) {
            ROS_ERROR("Unable to find matching keypoint for measurement??");
        }

        const auto& kp = keypoints_[kp_index];

        try {

            Eigen::MatrixXd Hpose, Hpoint;
            Eigen::Vector2d zhat =
              camera.project(kp->position(), Hpose, Hpoint);

            residuals.segment<2>(2 * kp_index) =
              msmt.keypoint_measurements[i].pixel_measurement - zhat;
            R_sqrt_vec.segment<2>(2 * kp_index) =
              msmt.keypoint_measurements[i].pixel_sigma *
              Eigen::Vector2d::Ones();

            // std::cout << "H size = " << H.rows() << " x " << H.cols() <<
            // std::endl; std::cout << "Hpoint size = " << Hpoint.rows() << " x
            // " << Hpoint.cols() << std::endl; std::cout << "Hpose size = " <<
            // Hpose.rows() << " x " << Hpose.cols() << std::endl; std::cout <<
            // "Hpose_compose size = " << Hpose_compose.rows() << " x " <<
            // Hpose_compose.cols() << std::endl; std::cout << "kp_index = " <<
            // kp_index << std::endl; std::cout << "x_index = " << x_index <<
            // std::endl;

            // Hl
            H.block<2, 3>(2 * kp_index, 3 * kp_index) = Hpoint;
            // Hq
            H.block<2, 3>(2 * kp_index, x_index) =
              Hpose * Hpose_compose.leftCols<4>() * Hquat_space;
            // Hp
            H.block<2, 3>(2 * kp_index, x_index + 3) =
              Hpose * Hpose_compose.rightCols<3>();

            n_observed++;

        } catch (CheiralityException& e) {
            // TODO should we ignore this or should we add a high value or
            // something
            residuals.segment<2>(2 * kp_index) = 100 * Eigen::Vector2d::Ones();
            H.block<2, 3>(2 * kp_index, 3 * kp_index) =
              Eigen::MatrixXd::Ones(2, 3);
            H.block<2, 6>(2 * kp_index, x_index) = Eigen::MatrixXd::Ones(2, 6);
        }
    }

    Eigen::MatrixXd R = R_sqrt_vec.array().pow(2).matrix().asDiagonal();

    double mahal = residuals.transpose() *
                   (H * Plx * H.transpose() + R).ldlt().solve(residuals);

    double factor = mahalanobisMultiplicativeFactor(2 * n_observed);

    // std::ofstream
    // out_file(fmt::format("/home/sean/code/object_pose_detection/debug_data/x{}_msmt{}_o{}.txt",
    //     keyframe->index(), msmt.global_msmt_id, id()));
    // out_file << "Camera pose: \n" <<
    // keyframe->pose().translation().transpose()
    // << std::endl; out_file << keyframe->pose().rotation().toRotationMatrix()
    // << std::endl; out_file << "keypoint positions: \n"; Eigen::MatrixXd
    // kp_positions(3, msmt.keypoint_measurements.size()); Eigen::MatrixXd
    // zhats(2, msmt.keypoint_measurements.size()); Eigen::MatrixXd kp_msmts(2,
    // msmt.keypoint_measurements.size()); Eigen::VectorXi
    // observed(msmt.keypoint_measurements.size()); for (size_t i = 0; i <
    // msmt.keypoint_measurements.size(); ++i) {
    //   int kp_index =
    //   findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id); if
    //   (kp_index < 0) {
    //     ROS_ERROR("Unable to find matching keypoint for measurement??");
    //   }

    //   const auto& kp = keypoints_[kp_index];

    //   try {
    //     zhats.col(kp_index) = camera.project(kp->position());
    //   } catch (CheiralityException& e) {
    //     zhats.col(kp_index).setZero();
    //   }

    //   kp_positions.col(kp_index) = kp->position();
    //   kp_msmts.col(kp_index) =
    //   msmt.keypoint_measurements[i].pixel_measurement; observed(kp_index) =
    //   msmt.keypoint_measurements[i].observed ? 1 : 0;
    // }
    // out_file << kp_positions << std::endl;
    // out_file << "KP measurements: \n" << kp_msmts << std::endl;
    // out_file << "zhats:\n" << zhats << std::endl;
    // out_file << "KP observed?:\n" << observed.transpose() << std::endl;
    // out_file << "final mahal = \n" << mahal << "\n factor = \n" << factor <<
    // std::endl; out_file << "Plx:\n" << Plx << "\nH:\n" << H << "\nR:\n" << R
    // << std::endl;

    // ROS_INFO_STREAM("Object " << id() << "; Mahal distance " << mahal
    //                           << " * factor " << factor << " = "
    //                           << mahal * factor);

    // right now if all the points are behind the camera (clearly wrong object)
    // the residuals vector will be all zero...
    if (n_observed == 0) {
        return std::numeric_limits<double>::max();
    }

    return mahal * factor;

    // for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
    //   if (!msmt.keypoint_measurements[i].observed)
    //     continue;

    //   const auto& kp_msmt = msmt.keypoint_measurements[i];

    //   int kp_match_index =
    //     findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id);
    //   if (kp_match_index >= 0) {
    //     if (keypoints_[kp_match_index]->initialized() &&
    //         !keypoints_[kp_match_index]->bad()) {

    //       Eigen::Vector2d zhat =
    //       camera.project(keypoints_[kp_match_index]->position());
    //       Eigen::Vector2d residual =
    //       msmt.keypoint_measurements[i].pixel_measurement - zhat; double
    //       px_sigma = msmt.keypoint_measurements[i].pixel_sigma;
    //       Eigen::Matrix2d R = px_sigma * px_sigma *
    //       Eigen::Matrix2d::Identity();

    //       Eigen::Matrix<double, 2, 9> H = computeProjectionJacobian(G_T_I,
    //                                                                 I_T_C_,
    //                                                                 keypoints_[kp_match_index]->position());

    //       Eigen::MatrixXd Plx =
    //       getPlx(sym::L(keypoints_[kp_match_index]->id()),
    //       Symbol(msmt.observed_key)); double d = residual.transpose() *
    //       (H*Plx*H.transpose() + R).lu().solve(residual);
    //       // double d =
    //       // keypoints_[kp_match_index]->computeMahalanobisDistance(kp_msmt);
    //       matched_distances.push_back(d);
    //     }
    //   }
    // }

    // if (matched_distances.size() == 0)
    //   return std::numeric_limits<double>::max();

    // double distance = 0.0;
    // for (auto& x : matched_distances)
    //   distance += x;

    // double factor = mahalanobisMultiplicativeFactor(2 *
    // matched_distances.size());

    // ROS_INFO_STREAM(" Mahal distance " << distance << " * factor " << factor
    // << " = " << distance * factor);

    // return distance * factor;
}

void
EstimatedObject::commitGraphSolution()
{
    pose_ = graph_pose_node_->pose();
    if (k_ > 0)
        basis_coefficients_ = graph_coefficient_node_->vector();

    for (auto& kp : keypoints_)
        kp->commitGraphSolution();
}

void
EstimatedObject::commitGtsamSolution(const gtsam::Values& values)
{
    pose_ = values.at<gtsam::Pose3>(sym::O(id_));
    if (k_ > 0)
        basis_coefficients_ = values.at<gtsam::Vector>(sym::C(id_));

    for (auto& kp : keypoints_)
        kp->commitGtsamSolution(values);
}

void
EstimatedObject::prepareGraphNode()
{
    graph_pose_node_->pose() = pose_;
    if (k_ > 0)
        graph_coefficient_node_->vector() = basis_coefficients_;

    for (auto& kp : keypoints_)
        kp->prepareGraphNode();
}

// double EstimatedObject::computeMeasurementLikelihood(const ObjectMeasurement&
// msmt) const
// {
// 	if (msmt.obj_name != obj_name_) return 0.0;

//     std::vector<double> matched_likelihoods;

//     // todo? can make this loop O(n log n) instead of O(n^2) by keeping a
//     sorted class index but not a big deal
//     for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {

//         if (!msmt.keypoint_measurements[i].observed) continue;

//         int kp_match_index =
//         findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id);
//         if (kp_match_index >= 0) {
//             // keypoint match

//             if (keypoints_[kp_match_index]->initialized() &&
//             !keypoints_[kp_match_index]->bad()) {
//                 double p =
//                 keypoints_[kp_match_index]->computeMeasurementLikelihood(msmt.keypoint_measurements[i]);

//                 matched_likelihoods.push_back(p);
//             }
//         }
//     }

//     if (matched_likelihoods.size() == 0) return 0.0;

//     // todo figure out best way to compute a final likelihood from
//     matched_likelihoods
//     // geometric mean
//     double likelihood = 1;
//     for (double l : matched_likelihoods) {
//         likelihood *= l;
//     }
//     likelihood = std::pow(likelihood, 1.0 / matched_likelihoods.size());

//     ROS_WARN_STREAM("Object " << id() << ", likelihood = " << likelihood);

//     return likelihood;
// }

std::vector<int64_t>
EstimatedObject::getKeypointIndices() const
{
    std::vector<int64_t> indices;
    for (const EstimatedKeypoint::Ptr& kp : keypoints_) {
        if (kp->inGraph())
            indices.push_back(kp->id());
    }
    return indices;
}

const std::vector<EstimatedKeypoint::Ptr>&
EstimatedObject::getKeypoints() const
{
    return keypoints_;
}

const std::vector<EstimatedKeypoint::Ptr>&
EstimatedObject::keypoints() const
{
    return keypoints_;
}

void
EstimatedObject::setIsVisible(boost::shared_ptr<SemanticKeyframe> kf)
{
    last_visible_ = kf->index();
}

void
EstimatedObject::addKeypointMeasurements(const ObjectMeasurement& msmt,
                                         double weight)
{
    // compute mahalanobis distance...
    if (in_graph_) {
        // if (measurements_.size() > 2) {
        double mahal_d = computeMahalanobisDistance(msmt);
        if (mahal_d < chi2inv95(2)) {
            ROS_INFO_STREAM("Adding measurement with mahal = " << mahal_d);
        } else {
            ROS_INFO_STREAM("Rejecting measurement with mahal = " << mahal_d);
            return;
        }
    }

    measurements_.push_back(msmt);

    auto keyframe = mapper_->getKeyframeByKey(msmt.observed_key);
    keyframe_observations_.push_back(keyframe);
    keyframe->visible_objects().push_back(shared_from_this());

    for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
        const KeypointMeasurement& kp_msmt = msmt.keypoint_measurements[i];

        if (!kp_msmt.observed)
            continue;

        int64_t local_kp_id = findKeypointByClass(kp_msmt.kp_class_id);

        if (local_kp_id < 0) {
            ROS_WARN_STREAM("Unable to find keypoint object for measurement");
            continue;
        }

        EstimatedKeypoint::Ptr kp = keypoints_[local_kp_id];

        if (kp->bad()) {
            continue;
        }

        modified_ = true;

        // ROS_INFO_STREAM("Initializing keypoint " << kp_msmt.kp_class_id);

        kp->addMeasurement(kp_msmt, weight);

        // last_seen_ = kp_msmt.pose_id;
    }

    last_seen_ = keyframe->index();

    // if (!in_graph_ && readyToAddToGraph()) {
    //   addToGraph();
    // }

    // TEST TODO
    if (!in_graph_) {
        optimizeStructure();
    }

    // if we're in the graph, update our local optimization problem without
    // actually solving it. we won't ever use the actual optimization but we
    // need it to contain all factors for local covariance information

    if (in_graph_) {
        structure_problem_->addCamera(keyframe, true);

        for (auto& kp_msmt : msmt.keypoint_measurements) {
            if (kp_msmt.observed)
                structure_problem_->addKeypointMeasurement(kp_msmt);
        }
    }
}

// void EstimatedObject::updateAndCheck(uint64_t pose_id, const gtsam::Values&
// estimate) {
//     update(pose_id, estimate);
// 	// sanityCheck(pose_id);
// }

void
EstimatedObject::updateGraphFactors()
{
    if (!inGraph())
        return;

    for (auto& kp : keypoints_) {
        kp->tryAddProjectionFactors();
    }
}

void
EstimatedObject::update(SemanticKeyframe::Ptr keyframe)
{
    if (bad())
        return;

    if (!params_.include_objects_in_graph &&
        keyframe->index() - last_seen_ < 10) {
        optimizeStructure();
    }

    // if (in_graph_) {
    //   ROS_INFO_STREAM("Object " << id() << " has graph pose " <<
    //   graph_pose_node_->pose());
    // }

    // sanityCheck(spur_node);

    if (!in_graph_ && keyframe->index() - last_seen_ > 10) {
        removeFromEstimation();
    }
}

Eigen::MatrixXd
EstimatedObject::getKeypointPositionMatrix() const
{
    // include *all* keypoints even those that are bad/poorly localized/etc
    Eigen::MatrixXd result(3, keypoints_.size());

    for (size_t i = 0; i < keypoints_.size(); ++i) {
        result.col(i) = keypoints_[i]->position();
    }

    return result;
}

Eigen::VectorXd
EstimatedObject::getKeypointOptimizationWeights() const
{
    // TODO figure out how best to do this
    Eigen::VectorXd w(keypoints_.size());

    // Inverse of measurement mahalanobis distances seems to be OK...
    // for (size_t i = 0; i < keypoints_.size(); ++i) {
    //     if (keypoints_[i]->nMeasurements() < 2) {
    //         w(i) = 0.0;
    //     } else {
    //         w(i) = 1 / keypoints_[i]->maxMahalanobisDistance(joint_marginal);
    //     }
    // }
    // w /= w.sum();

    for (size_t i = 0; i < keypoints_.size(); ++i) {
        w(i) = keypoints_[i]->detectionScoreSum();
    }

    return w;
}

bool
EstimatedObject::readyToAddToGraph()
{
    if (in_graph_)
        return false;

    // Count how many keypoints have enough measurements to be considered well
    // localized
    // TODO use a better metric than #measurements?
    size_t n_keypoints_localized = 0;
    for (const auto& kp : keypoints_) {
        if (kp->nMeasurements() > 2)
            n_keypoints_localized++;
    }

    if (n_keypoints_localized >= params_.min_object_n_keypoints) {
        return true;
    }

    return false;
}

void
EstimatedObject::addToGraph()
{
    // Compute a good initial estimate with a local optimization
    optimizeStructure();

    // catch (const std::exception& e)
    // {
    //   ROS_WARN_STREAM("Failed to optimize structure of object " << id());
    //   ROS_WARN_STREAM("Error: " << e.what());
    //   // removeFromEstimation();
    //   return;
    // }

    // Add our keypoints are in the graph
    for (EstimatedKeypoint::Ptr& kp : keypoints_) {
        kp->addToGraphForced();
    }

    modified_ = true;

    if (params_.include_objects_in_graph) {
        graph_->addNode(graph_pose_node_);
        if (k_ > 0)
            graph_->addNode(graph_coefficient_node_);
        graph_->addFactor(structure_factor_);

        semantic_graph_->addNode(graph_pose_node_);
        if (k_ > 0)
            semantic_graph_->addNode(graph_coefficient_node_);
        semantic_graph_->addFactor(structure_factor_);

        // graph_pose_node_->addToOrderingGroup(
        //   graph_->solver_options().linear_solver_ordering, 0);
        // if (k_ > 0)
        //     graph_coefficient_node_->addToOrderingGroup(
        //       graph_->solver_options().linear_solver_ordering, 1);

        in_graph_ = true;
    }
    // pose_added_to_graph_ = msmt.pose_id;

    ROS_INFO_STREAM("Object " << id() << " added to graph.");
}

void
EstimatedObject::setConstantInGraph()
{
    if (!in_graph_)
        return;

    graph_->setNodeConstant(graph_pose_node_);
    if (k_ > 0)
        graph_->setNodeConstant(graph_coefficient_node_);

    semantic_graph_->setNodeConstant(graph_pose_node_);
    if (k_ > 0)
        semantic_graph_->setNodeConstant(graph_coefficient_node_);

    for (auto& kp : keypoints_) {
        kp->setConstantInGraph();
    }
}

void
EstimatedObject::setVariableInGraph()
{
    if (!in_graph_)
        return;

    graph_->setNodeVariable(graph_pose_node_);
    if (k_ > 0)
        graph_->setNodeVariable(graph_coefficient_node_);

    semantic_graph_->setNodeVariable(graph_pose_node_);
    if (k_ > 0)
        semantic_graph_->setNodeVariable(graph_coefficient_node_);

    for (auto& kp : keypoints_) {
        kp->setVariableInGraph();
    }
}

void
EstimatedObject::removeFromEstimation()
{
    is_bad_ = true;

    ROS_WARN_STREAM("Removing object " << id() << " from estimation.");

    for (auto& kp : keypoints_) {
        kp->removeFromEstimation();
    }

    if (params_.include_objects_in_graph) {
        graph_->removeNode(graph_pose_node_);
        if (k_ > 0)
            graph_->removeNode(graph_coefficient_node_);
        graph_->removeFactor(structure_factor_);

        semantic_graph_->removeNode(graph_pose_node_);
        if (k_ > 0)
            semantic_graph_->removeNode(graph_coefficient_node_);
        semantic_graph_->removeFactor(structure_factor_);
    }

    in_graph_ = false;
}

// void
// EstimatedObject::sanityCheck(NodeInfoConstPtr latest_spur_node)
// {
//   if (bad())
//     return;

//   int64_t latest_idx = static_cast<int64_t>(latest_spur_node->index());
//   int64_t last_seen = static_cast<int64_t>(last_seen_);

//   // ROS_INFO_STREAM(fmt::format("Last seen: {}, latest index: {}",
//   last_seen,
//   // latest_idx));

//   if (!in_graph_ && last_seen <= latest_idx - 10) {
//     // ROS_INFO("Here!!");
//     // ROS_INFO_STREAM("Last seen: " << last_seen);
//     // ROS_INFO_STREAM("index - 10 = " << latest_idx - 10);
//     removeFromEstimation();
//     return;
//   }

//   if (!in_graph_)
//     return;

//   if (checkObjectExploded()) {
//     ROS_WARN_STREAM("Object " << id_ << " poorly localized... ");

//     removeFromEstimation();

//     // size_t insane_steps = 0;
//     // size_t limit = 10;

//     // while (insane_steps < limit && checkObjectExploded()) {
//     //     graph_->isamUpdate();
//     //     update(pose_id, graph_->calculateBestEstimate());
//     //     insane_steps++;
//     //     ROS_WARN_STREAM("Trying to recover, step " << insane_steps << "/"
//     <<
//     //     limit);
//     // }

//     // if (insane_steps >= limit) {
//     //     ROS_INFO_STREAM("Marking object " << id_ << " as bad due to poor
//     //     localization.");
//     //     // removeFromEstimation();
//     // } else {
//     //     ROS_INFO_STREAM("Object " << id_ << " recovered.");
//     // }
//   }
// }

// bool
// EstimatedObject::checkObjectExploded() const
// {
//   bool any_bad = false;

//   for (const EstimatedKeypoint::Ptr& kp : keypoints_) {
//     double d_from_centroid = pose_.range(kp->position());
//     // double d_structure = gtsam::Pose3::identity().range(kp->structure());

//     // ROS_WARN_STREAM("Obj " << i << " kp " <<
//     object_keypoint_indices_[i][j]
//     // << " distance = " << d_from_centroid);
//     if (d_from_centroid > 10) {
//       any_bad = true;
//       break;
//     }
//   }

//   return any_bad;
// }

size_t
EstimatedObject::countObservedFeatures(const ObjectMeasurement& msmt) const
{
    size_t n_observed = 0;
    for (const KeypointMeasurement& kp_msmt : msmt.keypoint_measurements) {
        if (kp_msmt.observed)
            n_observed++;
    }
    return n_observed;
}
