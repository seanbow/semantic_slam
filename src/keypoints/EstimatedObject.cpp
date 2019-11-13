
#include "semantic_slam/keypoints/EstimatedObject.h"
// #include "omnigraph/keypoints/StructureFactor.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"

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
  boost::shared_ptr<FactorGraph> graph, const ObjectParams& params,
  geometry::ObjectModelBasis object_model, uint64_t object_id,
  uint64_t first_keypoint_id, const ObjectMeasurement& msmt,
  const Pose3& G_T_C, const Pose3& I_T_C, std::string platform,
  boost::shared_ptr<CameraCalibration> calibration)
  : graph_(graph)
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
{
  // NOTE : because we need shared_from_this() to create the child keypoints,
  //  we cannot call the initialization method from within the constructor.
  // But also it doesn't make sense to have such an uninitialized object.
  // --> Make this constructor private and require the use of a create() method.
  // initializeFromMeasurement(msmt, G_T_C);

  // std::string
  // model_path("/home/sean/rcta/src/mapping/semslam/semslam/models/car_basis.dat");
  // model_ = geometry::readModelFile(model_path);

  m_ = model_.mu.cols();
  k_ = model_.pc.rows() / 3;

  last_visible_ = Symbol(msmt.observed_key).index();

  graph_pose_node_ = util::allocate_aligned<SE3Node>(sym::O(id_));

  if (k_ > 0) {
    graph_coefficient_node_ = util::allocate_aligned<VectorXdNode>(sym::C(id_), 
                                                                   boost::none, // no associated time
                                                                   k_);
    graph_coefficient_node_->vector().setZero();
  }
}

EstimatedObject::Ptr
EstimatedObject::create(boost::shared_ptr<FactorGraph> graph,
                        const ObjectParams& params,
                        geometry::ObjectModelBasis object_model,
                        uint64_t object_id, uint64_t first_keypoint_id,
                        const ObjectMeasurement& msmt,
                        const Pose3& G_T_C, const Pose3& I_T_C,
                        std::string platform,
                        boost::shared_ptr<CameraCalibration> calibration)
{
  EstimatedObject::Ptr pt(new EstimatedObject(
    graph, params, object_model, object_id, first_keypoint_id, msmt, G_T_C,
    I_T_C, platform, calibration));
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
  graph_pose_node_->pose() = G_T_C.compose(C_T_O);

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
        new EstimatedKeypoint(graph_, params_, first_kp_id_ + i, id_,
                              msmt.keypoint_measurements[i].kp_class_id, I_T_C_,
                              platform_, camera_calibration_, shared_this));
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
  Eigen::VectorXd w = params_.structure_error_coefficient * Eigen::VectorXd::Ones(keypoints_.size());

  structure_factor_ = util::allocate_aligned<CeresStructureFactor>(
    graph_pose_node_,
    keypoint_nodes,
    graph_coefficient_node_,
    model_,
    w,
    params_.structure_regularization_factor
  );

}

bool
EstimatedObject::inGraph() const
{
  return in_graph_;
}

// std::vector<Key>
// EstimatedObject::getKeypointKeys() const
// {
//   std::vector<Key> keys;

//   for (size_t i = 0; i < keypoints_.size(); ++i) {
//     keys.push_back(sym::L(keypoints_[i]->id()));
//   }

//   return keys;
// }

// std::vector<Key>
// EstimatedObject::getKeypointKeysInGraph() const
// {
//   std::vector<Key> keys;

//   for (size_t i = 0; i < keypoints_.size(); ++i) {
//     if (keypoints_[i]->inGraph())
//       keys.push_back(sym::L(keypoints_[i]->id()));
//   }

//   return keys;
// }

// std::vector<Key>
// EstimatedObject::getAllKeys() const
// {
//   std::vector<Key> keys = getKeypointKeys();
//   keys.push_back(sym::O(id_));
//   return keys;
// }

// std::vector<Key>
// EstimatedObject::getKeysInGraph() const
// {
//   std::vector<Key> keys = getKeypointKeysInGraph();
//   if (inGraph())
//     keys.push_back(sym::O(id_));
//   return keys;
// }

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

Pose3
EstimatedObject::pose() const
{
  return graph_pose_node_->pose();
}

void
EstimatedObject::optimizeStructure()
{
  size_t m = model_.mu.cols();
  size_t k = model_.pc.rows() / 3;

  // structure_graph_ = gtsam::NonlinearFactorGraph();
  // structure_optimization_values_.clear();

  Eigen::MatrixXd mu = geometry::centralize(model_.mu);
  Eigen::MatrixXd pc = geometry::centralize(model_.pc);

  double lambda = params_.structure_regularization_factor;

  // ROS_WARN_STREAM("Beginning structure optimization, m = " << m << "; k = " << k);
  // ROS_WARN_STREAM("Object " << id_ << ", name " << obj_name());


  Eigen::VectorXd weights = Eigen::VectorXd::Ones(m);
  Pose3 body_T_camera(I_T_C_.rotation(),
                      I_T_C_.translation());

  structure_problem_ = boost::make_shared<StructureOptimizationProblem>(model_, *camera_calibration_,
                                                    body_T_camera, weights, params_);

  Pose3 initial_object_pose = graph_pose_node_->pose();

  structure_problem_->initializePose(initial_object_pose);

  for (auto& obj_msmt : measurements_) {
    auto cam_node = graph_->getNode<SE3Node>(obj_msmt.observed_key);
    if (!cam_node) {
      ROS_ERROR_STREAM("Unable to find graph node for camera pose " << DefaultKeyFormatter(obj_msmt.observed_key));
    }

    Pose3 cam_pose = cam_node->pose();

    // gtsam::Pose3 cam_pose;
    // graph_->getSolution(obj_msmt.observed_symbol, cam_pose);
    // Pose3 pose(cam_pose.rotation(),
    //            cam_pose.translation().vector());

    // TODO should we be using the real covariances here?? or at least in
    // covariance computation within ceres?

    // Eigen::Matrix<double, 6, 6> prior_noise =
    //   1.0e-6 * Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 1> prior_noise_vec;
    prior_noise_vec << 1e-3 * Eigen::Vector3d::Ones(), 
                       1e-2 * Eigen::Vector3d::Ones();

    structure_problem_->addCameraPose(Symbol(obj_msmt.observed_key).index(), 
                                      cam_pose, 
                                      prior_noise_vec.asDiagonal(),
                                      false);
  }

  for (auto& kp : keypoints_) {
    structure_problem_->initializeKeypointPosition(kp->classid(), kp->position());

    int n_meas = 0;

    for (auto& kp_msmt : kp->measurements()) {
      if (kp_msmt.observed) {
        structure_problem_->addKeypointMeasurement(kp_msmt);
        n_meas++;
      }
    }

    // std::cout << "Keypoint " << kp->id() << " has " << n_meas << " measurements." << std::endl;
  }

  structure_problem_->solve();

  Pose3 ceres_pose = structure_problem_->getObjectPose();

  graph_pose_node_->pose() = ceres_pose;

  if (k > 0)
    basis_coefficients_ = structure_problem_->getBasisCoefficients();

  for (auto& kp : keypoints_) {
    if (!kp->bad())
      kp->setPosition(*structure_problem_->getKeypoint(kp->classid()));
  }

  // test get keypoint marginals
  // for (auto& kp : keypoints_) {
  //   auto Plx = structure_problem_->getPlx(kp->classid(), measurements_.back().observed_symbol.index());
  //   std::cout << "Plx for id " << kp->classid() << " is\n" << Plx << std::endl;
  // }

  // compute keypoint marginals!!
  // THIS is no longer needed here -- computation done by and retrieved from the ceres optimization
  // problem directly
  // if (!params_.include_objects_in_graph) {
  //   for (auto& kp : keypoints_) {
  //     if (!kp->bad()) {
  //       kp->setGlobalCovariance(1e-2 * Eigen::Matrix3d::Identity());
  //     }
  //   }
  // }

  // std::cout << "Optimization result -->\n" << pose_ << std::endl;
  // std::cout << "c = " << basis_coefficients_.transpose() << std::endl;
}

// double
// EstimatedObject::getStructureError() const
// {
//   return structure_graph_.error(structure_optimization_values_);
// }


Eigen::MatrixXd
EstimatedObject::getPlx(Key l_key, Key x_key)
{
  // TODO TODO!
  // Eigen::Matrix<double, 9, 9> cov = Eigen::Matrix<double, 9, 9>::Zero();
  // cov.block<3,3>(6,6) = 0.05 * Eigen::Matrix3d::Identity();
  // return cov;

  if (!structure_problem_) {
    throw std::logic_error("Tried to extract covariance information before structure optimization.");
  }

  size_t kp_match_index = findKeypointByKey(l_key);
  const auto& kp = keypoints_[kp_match_index];

  Symbol x_symbol(x_key);

  Eigen::MatrixXd Plx = structure_problem_->getPlx(kp->classid(), x_symbol.index());

  return Plx;
}

double
EstimatedObject::computeMahalanobisDistance(const ObjectMeasurement& msmt) const
{
  // if (msmt.class_id != classid_) return std::numeric_limits<double>::max();
  // ROS_WARN_STREAM("Computing mahal for object " << id());
  if (msmt.obj_name != obj_name_) {
    // ROS_WARN_STREAM("Object id " << id() << " name " << obj_name_
    //                              << " != measurement class " << msmt.obj_name);
    return std::numeric_limits<double>::max();
  }

  std::vector<double> matched_distances;

  for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
    if (!msmt.keypoint_measurements[i].observed)
      continue;

    const auto& kp_msmt = msmt.keypoint_measurements[i];

    int kp_match_index =
      findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id);
    if (kp_match_index >= 0) {
      if (keypoints_[kp_match_index]->initialized() &&
          !keypoints_[kp_match_index]->bad()) {
        double d =
          keypoints_[kp_match_index]->computeMahalanobisDistance(kp_msmt);
        matched_distances.push_back(d);
      }
    }
  }

  if (matched_distances.size() == 0)
    return std::numeric_limits<double>::max();

  double distance = 0.0;
  for (auto& x : matched_distances)
    distance += x;

  double factor = mahalanobisMultiplicativeFactor(2 * matched_distances.size());

  // ROS_INFO_STREAM(" Mahal distance " << distance << " * factor " << factor <<
  // " = " << distance * factor);

  return distance * factor;
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

void
EstimatedObject::addKeypointMeasurements(const ObjectMeasurement& msmt,
                                         double weight)
{
  measurements_.push_back(msmt);

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

  last_seen_ = Symbol(msmt.observed_key).index();

  if (!in_graph_) {
    tryAddSelfToGraph(msmt);
  }

  // TEST TODO
  if (!params_.include_objects_in_graph) {
    optimizeStructure();
  }
}

// void EstimatedObject::updateAndCheck(uint64_t pose_id, const gtsam::Values&
// estimate) {
//     update(pose_id, estimate);
// 	// sanityCheck(pose_id);
// }

void
EstimatedObject::update(CeresNodePtr spur_node)
{
  if (bad())
    return;

  if (in_graph_ && !params_.include_objects_in_graph &&
      spur_node->index() - last_seen_ < 10) {
    // TIME_TIC;
    optimizeStructure();
    // TIME_TOC(fmt::format("Structure optimization for object {}", id()));
  }

  if (in_graph_ && params_.include_objects_in_graph) {
    ROS_INFO_STREAM("Object " << id() << " has graph pose " << graph_pose_node_->pose());
  }

  for (EstimatedKeypoint::Ptr& kp : keypoints_) {
    if (!kp->bad())
      kp->update();
  }

  // sanityCheck(spur_node);

  // if (!in_graph_ && pose_id - last_seen_ > 10) {
  //     removeFromEstimation();
  // }
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

void
EstimatedObject::tryAddSelfToGraph(const ObjectMeasurement& msmt)
{
  if (in_graph_)
    return;

  // Count how many keypoints are in the graph
  size_t n_keypoints_localized = 0;

  for (auto kp : keypoints_) {
    if (kp->inGraph())
      n_keypoints_localized++;
    // if (kp->nMeasurements() > 2) n_keypoints_localized++;
  }

  // ROS_INFO_STREAM("Object " << id() << " has " << n_keypoints_localized << "
  // localized keypoints.");

  if (n_keypoints_localized >= params_.min_object_n_keypoints) {
    // add self to the graph

    // Update our structure given the keypoints we have so far.
    try {
      optimizeStructure();
    }
    catch (const std::exception& e)
    {
      ROS_WARN_STREAM("Failed to optimize structure of object " << id());
      ROS_WARN_STREAM("Error: " << e.what());
      // removeFromEstimation();
      return;
    }

    // Make sure all of our keypoints are in the graph
    for (EstimatedKeypoint::Ptr& kp : keypoints_) {
      kp->addToGraphForced();
    }

    modified_ = true;

    // graph_->addNode(graph_pose_node_);
    // if (k_ > 0) graph_->addNode(graph_coefficient_node_);
    // graph_->addFactor(structure_factor_);

    in_graph_ = true;
    // pose_added_to_graph_ = msmt.pose_id;

    ROS_INFO_STREAM("Object " << id() << " added to graph.");
  }
}

void
EstimatedObject::removeFromEstimation()
{
  is_bad_ = true;

  ROS_WARN_STREAM("Removing object " << id() << " from estimation.");

  // return; // TODO why does everything below this cause a crash??? --> it was
  // a GTSAM bug

  for (auto& kp : keypoints_) {
    kp->removeFromEstimation();
  }

  if (params_.include_objects_in_graph) {
    throw std::logic_error("not implemented");
  }

  // graph_->isamUpdate({}, gtsam::Values(), getFactorIndicesForRemoval());

  in_graph_ = false;
}

// void
// EstimatedObject::sanityCheck(NodeInfoConstPtr latest_spur_node)
// {
//   if (bad())
//     return;

//   int64_t latest_idx = static_cast<int64_t>(latest_spur_node->index());
//   int64_t last_seen = static_cast<int64_t>(last_seen_);

//   // ROS_INFO_STREAM(fmt::format("Last seen: {}, latest index: {}", last_seen,
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
//     //     ROS_WARN_STREAM("Trying to recover, step " << insane_steps << "/" <<
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

//     // ROS_WARN_STREAM("Obj " << i << " kp " << object_keypoint_indices_[i][j]
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
