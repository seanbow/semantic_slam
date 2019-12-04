#include "semantic_slam/keypoints/EstimatedKeypoint.h"
// #include "semslam/JSON.h"
#include "semantic_slam/keypoints/EstimatedObject.h"
#include "semantic_slam/Utils.h"

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"

// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/geometry/CameraSet.h>
// #include <gtsam/geometry/triangulation.h>

#include <fmt/format.h>
#include <fstream>
#include <iostream>

#include "semantic_slam/Camera.h"
#include "semantic_slam/SemanticMapper.h"

namespace sym = symbol_shorthand;

EstimatedKeypoint::EstimatedKeypoint(boost::shared_ptr<FactorGraph> graph, const ObjectParams& params, size_t id,
                                     size_t object_id, size_t class_id, Pose3 I_T_C, std::string platform,
                                     boost::shared_ptr<CameraCalibration> camera_calib, EstimatedObject::Ptr parent,
                                     SemanticMapper* mapper)
  : graph_(graph)
  , params_(params)
  , global_id_(id)
  , object_id_(object_id)
  , classid_(class_id)
  , I_T_C_(I_T_C)
  , platform_(platform)
  , camera_calibration_(camera_calib)
  , in_graph_(false)
  , is_bad_(false)
  ,
  //  object_in_graph_(false),
  initialized_(false)
  , detection_score_sum_(0.0)
  , parent_(parent),
  mapper_(mapper)
{
  // initializeFromMeasurement(msmt);
  graph_node_ = util::allocate_aligned<Vector3dNode>(sym::L(id));
}

void EstimatedKeypoint::commitGraphSolution()
{
  global_position_ = graph_node_->vector();
}


void EstimatedKeypoint::commitGtsamSolution(const gtsam::Values& values)
{
  global_position_ = values.at<gtsam::Point3>(graph_node_->key());
}

void EstimatedKeypoint::prepareGraphNode()
{
  graph_node_->vector() = global_position_;
}

void EstimatedKeypoint::addMeasurement(const KeypointMeasurement& msmt, double weight)
{
  // if we haven't been initialized yet do so from this first measurement
  if (!initialized_)
  {
    ROS_WARN_STREAM("[EstimatedKeypoint] Keypoint uninitialized! Initializing from first added measurement");
    initializeFromMeasurement(msmt);
  }

  // Check if it's reasonable
  // TODO check structure/consistency errors rather than just mahal distance?

  // if this is the first measurement our position will have been initialized for it so any sort of
  // error checking is pointless
  if (measurements_.size() > 0) {
    double mahal_d = computeMahalanobisDistance(msmt);
    if (mahal_d > chi2inv95(2)) {
      // ROS_WARN_STREAM(fmt::format("REJECTED measurement of keypoint {}, mahal = {}, score = {}", 
      //                             id(), mahal_d, msmt.score));
      return;
    }

    // ROS_INFO_STREAM(fmt::format("Adding measurement of keypoint {} with mahal dist {}, score = {}", 
    //                             id(), mahal_d, msmt.score));
  }

  measurements_.push_back(msmt);
  detection_score_sum_ += msmt.score;

  // ROS_INFO_STREAM("Adding measurement to keypoint " << id());

  Eigen::Vector2d noise_vec = Eigen::Vector2d::Constant(msmt.pixel_sigma);
  auto camera_node = mapper_->keyframes()[Symbol(msmt.measured_key).index()]->graph_node();
  // auto camera_node = graph_->getNode<SE3Node>(msmt.measured_key);
  CeresProjectionFactorPtr proj_factor = util::allocate_aligned<CeresProjectionFactor>(
      camera_node,
      graph_node_,
      msmt.pixel_measurement,
      noise_vec.asDiagonal(),
      camera_calibration_,
      I_T_C_
  );

  projection_factors_.push_back(proj_factor);

  measurement_weights_.push_back(weight);

  // TODO "safety" check / error check
  // TODO factor uniqueness check???
  // if (in_graph_)
  // {
  //   tryAddProjectionFactors();
  // }

  // last_seen_ = msmt.pose_id;
}

double EstimatedKeypoint::measurementWeightSum() const
{
  double sum = 0.0;
  for (double w : measurement_weights_)
    sum += w;
  return sum;
}

void EstimatedKeypoint::initializeFromMeasurement(const KeypointMeasurement& msmt)
{
  initializePosition(msmt);

  initialized_ = true;
}

void EstimatedKeypoint::removeFromEstimation()
{
  is_bad_ = true;

  if (!in_graph_)
    return;

  ROS_WARN_STREAM("Removing kp " << id() << " from estimation.");

  if (params_.include_objects_in_graph)
  {
    for (auto factor : projection_factors_) {
      graph_->removeFactor(factor);
    }

    graph_->removeNode(graph_node_);
  }

  in_graph_ = false;
}

void EstimatedKeypoint::initializePosition(const KeypointMeasurement& msmt)
{
  // Initialize a point along the camera ray some distance from the camera with
  // a high covariance along the ray
  Eigen::Vector3d z_unit;
  z_unit << msmt.normalized_measurement, 1;
  z_unit.normalize();

  double d_init = msmt.depth;

  Eigen::Vector3d C_l = d_init * z_unit;

  // auto node = graph_->getNode<SE3Node>(msmt.measured_key);
  // Pose3 G_x_C = node->pose();
  Pose3 G_x_I = mapper_->keyframes()[Symbol(msmt.measured_key).index()]->pose();
  Pose3 G_x_C = G_x_I.compose(I_T_C_);

  Eigen::Vector3d G_l = G_x_C.transform_from(C_l);

  // Compute covariance
  double sigma_xy = d_init * msmt.pixel_sigma / camera_calibration_->fx();
  double sigma_z = msmt.depth_sigma;

  Eigen::Matrix3d P_local = Eigen::Vector3d(sigma_xy, sigma_xy, sigma_z).array().pow(2).matrix().asDiagonal();
  // Rotate into global frame
  Eigen::Matrix3d P = G_x_C.rotation().toRotationMatrix() * P_local * G_x_C.rotation().toRotationMatrix().transpose();

  // ROS_WARN_STREAM("P_local = " << P_local.diagonal().transpose());
  // ROS_WARN_STREAM("Initialized P as:\n" << P);

  global_position_ = G_l;
  // graph_node_->vector() = G_l;
  global_covariance_ = P;
}

bool EstimatedKeypoint::triangulate(boost::optional<double&> condition)
{
  if (measurements_.size() < 2)
    return false;

  // ROS_INFO_STREAM("Triangulation of landmark " << id() << ". Pre-triang position = " << position());

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * measurements_.size(), measurements_.size() + 3);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * measurements_.size(), 1);

  Eigen::Vector3d p_meas;
  p_meas(2) = 1.0;

  aligned_vector<Pose3> msmt_poses;

  for (size_t i = 0; i < measurements_.size(); ++i)
  {
    // auto node = graph_->getNode<SE3Node>(measurements_[i].measured_key);
    // Pose3 G_x_I = node->pose();
    Pose3 G_x_I = mapper_->keyframes()[Symbol(measurements_[i].measured_key).index()]->pose();
    msmt_poses.push_back(G_x_I.compose(I_T_C_));

    // msmt_poses.push_back(graph_->calculateEstimate<gtsam::Pose3>(sym::X(measurements_[i].pose_id)).compose(I_T_C_));

    p_meas.head<2>() = camera_calibration_->calibrate(measurements_[i].pixel_measurement);

    A.block<3, 3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3, 1>(3 * i, 3 + i) = -(msmt_poses[i].rotation() * p_meas);
    b.block<3, 1>(3 * i, 0) = msmt_poses[i].translation();
  }

  // Solve least squares system

  Eigen::Vector3d point;

  // If the condition number is requested we have to do SVD of A anyway so use it to
  // solve the system as well
  if (condition)
  {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    point = svd.solve(b).block<3, 1>(0, 0);
    *condition = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  }
  else
  {
    // QR should be stable enough
    point = A.colPivHouseholderQr().solve(b).block<3, 1>(0, 0);
  }

  // ROS_INFO_STREAM("Triangulation result = " << point);

  // check that the point is in front of all cameras
  bool is_good = true;
  for (size_t i = 0; i < measurements_.size(); ++i)
  {
    const Eigen::Vector3d& p_local = msmt_poses[i].transform_to(point);

    // ROS_WARN_STREAM("Triang result locally = " << p_local);
    if (p_local(2) <= 0)
    {
      is_good = false;
      break;
    }
  }

  if (is_good)
  {
    // graph_node_->vector() = point;

    // estimate covariance
    double msmt_noise_sig2 = measurements_[0].pixel_sigma * measurements_[0].pixel_sigma;
    msmt_noise_sig2 /= (camera_calibration_->fx() * camera_calibration_->fy());

    global_covariance_ = msmt_noise_sig2 * (A.transpose() * A).inverse().block<3, 3>(0, 0);
  }

  return is_good;
}

double EstimatedKeypoint::totalMahalanobisError() const
{
  double distance = 0;

  for (auto msmt : measurements_)
  {
    distance += computeMahalanobisDistance(msmt);
  }

  return distance;
}

size_t EstimatedKeypoint::nMeasurements() const
{
  // return measurements_.size();
  
  // Return only the number of measurements whose keyframes are in the graph
  size_t count = 0;

  for (const auto& msmt : measurements_) {
    const auto& kf = mapper_->getKeyframeByKey(msmt.measured_key);
    if (kf->inGraph()) count++;
  }

  return count;
}

double EstimatedKeypoint::maxMahalanobisDistance() const
{
  double max_err = 0;
  for (auto msmt : measurements_)
  {
    double this_err = computeMahalanobisDistance(msmt);
    if (this_err > max_err)
    {
      max_err = this_err;
    }
  }

  return max_err;
}

double EstimatedKeypoint::mahalDistanceThreshold() const
{
  // Distance ~ Chi2(2 * residuals.size())

  // count number of depth measurements
  int n_depth = 0;
  for (auto& meas : measurements_)
  {
    if (meas.measured_depth > 0)
      n_depth++;
  }

  return chi2inv95(2 * measurements_.size() + n_depth);
}

bool EstimatedKeypoint::checkSafeToAdd()
{
  // TODO TODO TODO
  return true;

  // ROS_WARN_STREAM("      > Checking if object " << parent_->id() << " keypoint " << id() << " safe to add");

  bool triangulation_succeeded = true;
  double cond = 0;

  if (params_.include_objects_in_graph)
  {
    triangulation_succeeded = triangulate(cond);

    if (!triangulation_succeeded)
    {
      ROS_WARN_STREAM("-- Rejected landmark " << id() << ", triangulation failed (cond = " << cond << ")");
      return false;
    }
  }

  // mahalanobis gate

  double distance = totalMahalanobisError();
  // ROS_WARN_STREAM("       > Mahal dist = " << distance);
  // ROS_WARN_STREAM("       > Triangulation cond = " << cond);

  double thresh = mahalDistanceThreshold();

  if (params_.include_objects_in_graph)
  {
    if (distance < thresh && cond < 1e4)
    {
      ROS_INFO_STREAM("-- Accepted landmark " << id() << ", mahal = " << distance << ", cond = " << cond);
      return true;
    }
    else
    {
      ROS_WARN_STREAM("-- Rejected landmark " << id() << ", mahal = " << distance << ", cond = " << cond);
      return false;
    }
  }
  else
  {
    if (distance < thresh)
    {
      ROS_INFO_STREAM(fmt::format("-- Accepted landmark {}, mahal {} < {} threshold", id(), distance, thresh));
      return true;
    }
    else
    {
      ROS_WARN_STREAM(fmt::format("-- Rejected landmark {}, mahal {} >= {} threshold", id(), distance, thresh));
      return false;
    }
  }
}

void EstimatedKeypoint::addToGraphForced()
{
  // add to graph no matter what
  // if the safety checks for the current measurements fail throw them out and add a bare point
  // this should only be called by an object which will then add a constraint tying this to
  // an object structure, otherwise the system will become indeterminate

  if (in_graph_)
    return;

  // ROS_INFO_STREAM("Adding keypoint " << id() << " to graph [forced].");

  // note -- if we only have one measurement it's "unsafe" to add this alone as
  // the triangulation is underconstrained. but if we're adding it with object
  // structure then that is still ok.
  // so mark safe if measurements_.size() <= 1.
  bool unsafe_to_add = measurements_.size() > 1 && !checkSafeToAdd();

  if (params_.include_objects_in_graph && (is_bad_ || unsafe_to_add))
  {
    // remove all measurements of this point
    // things like the id, calibration, etc remain the same of course
    // projection_factors_.clear();
    // depth_factors_.clear();
    measurement_weights_.clear();
    measurements_.clear();
  }

  is_bad_ = false;

  if (params_.include_objects_in_graph) {
    // proceed to add to graph as normal
    graph_->addNode(graph_node_);

    tryAddProjectionFactors();
    // graph_->addFactors(projection_factors_);
  
    in_graph_ = true;
  }
}

void EstimatedKeypoint::tryAddProjectionFactors()
{
  // Only add factors whose associated keyframe is in the graph
  for (int i = 0; i < measurements_.size(); ++i) {
    const auto& kf = mapper_->getKeyframeByKey(measurements_[i].measured_key);
    if (kf->inGraph() && !projection_factors_[i]->active()) {
      graph_->addFactor(projection_factors_[i]);
    }
  }
}

void EstimatedKeypoint::addToGraph()
{
  if (in_graph_ || is_bad_)
    return;

  if (checkSafeToAdd())
  {
    ROS_INFO_STREAM("Adding keypoint " << id() << " to graph.");

    if (params_.include_objects_in_graph) {
      graph_->addNode(graph_node_);
      graph_->addFactors(projection_factors_);

      in_graph_ = true;
    }
  }
  else
  {
    is_bad_ = true;
  }
}

void EstimatedKeypoint::setConstantInGraph()
{
  graph_->setNodeConstant(graph_node_);
}

void EstimatedKeypoint::setVariableInGraph()
{
  graph_->setNodeVariable(graph_node_);
}

std::vector<Key> EstimatedKeypoint::getObservedKeys() const
{
  std::vector<Key> keys;
  for (auto& msmt : measurements_)
  {
    keys.push_back(msmt.measured_key);
  }
  return keys;
}

double EstimatedKeypoint::computeMahalanobisDistance(const KeypointMeasurement& msmt) const
{
  if (msmt.kp_class_id != classid_)
  {
    ROS_WARN_STREAM("Mahalanobis distance called for landmark " << id() << "with wrong class");
    return std::numeric_limits<double>::max();
  }

  if (is_bad_)
  {
    ROS_WARN_STREAM("Mahalanobis distance called for BAD landmark " << id());
    return std::numeric_limits<double>::max();
  }

  auto keyframe = mapper_->keyframes()[Symbol(msmt.measured_key).index()];
  Pose3 G_T_I = keyframe->pose();

  // Pose3 G_T_I = graph_->getNode<SE3Node>(msmt.measured_key)->pose();

  Camera camera(G_T_I.compose(I_T_C_));
  Eigen::Vector2d zhat;

  try
  {
    zhat = camera.project(position());
  }
  catch (CheiralityException& e)
  {
    return std::numeric_limits<double>::max();
  }

  // ROS_WARN_STREAM("Computing mahal for object " << parent_->id() << "; kp " << id());

  // check if visible
  // if (fabs(zhat(0)) > 1.25 || fabs(zhat(1)) > 1.25) {
  //     // clearly not in frame
  //     return std::numeric_limits<double>::max();
  // }

  Eigen::Vector2d residual = msmt.normalized_measurement - zhat;

  // Eigen::Matrix<double, 2, 9> H = computeProjectionJacobian(G_T_I.rotation().toRotationMatrix(), 
  //                                                           G_T_I.translation(),
  //                                                           I_T_C_.rotation().toRotationMatrix(), 
  //                                                           position());
  Eigen::Matrix<double, 2, 9> H = computeProjectionJacobian(G_T_I, I_T_C_, position());

  Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
  double px_sigma = msmt.pixel_sigma;
  R(0, 0) = px_sigma * px_sigma / (camera_calibration_->fx() * camera_calibration_->fx());
  R(1, 1) = px_sigma * px_sigma / (camera_calibration_->fy() * camera_calibration_->fy());

  // std::cout << "R = \n" << R << std::endl;
  // std::cout << "px sigma = " << px_sigma << "; fx = " << camera_calibration_->fx() << std::endl;

  // Eigen::Matrix2d S = H * Plx * H.transpose() + R;
  // double mahal = residual.transpose() * S.inverse() * residual;

  Eigen::MatrixXd Plx_full = parent_->getPlx(sym::O(parent_->id()), Symbol(msmt.measured_key));

  // this is dumb
  // figure out which index we are in Plx...
  int our_index = 0;
  for (int i = 0; i < parent_->keypoints().size(); ++i) {
    if (parent_->keypoints()[i]->id() == id()) {
      our_index = i;
      break;
    }
  }

  size_t x_index = 3 * parent_->keypoints().size();
  Eigen::MatrixXd Plx(9,9);
  Plx.topLeftCorner<3,3>() = Plx_full.block<3,3>(3*our_index, 3*our_index);
  Plx.bottomRightCorner<6,6>() = Plx_full.block<6,6>(x_index, x_index);
  Plx.block<3,6>(0,3) = Plx_full.block<3,6>(3*our_index, x_index);
  Plx.block<6,3>(3,0) = Plx_full.block<6,3>(x_index, 3*our_index);


  // std::cout << "Plx for landmark " << id() << ": " << std::endl;
  // std::cout << " in graph?: " << in_graph_ << "\n";
  // std::cout << Plx << std::endl;

  double mahal = residual.transpose() * (H * Plx * H.transpose() + R).lu().solve(residual);

  return mahal;
}

// double EstimatedKeypoint::computeMeasurementLikelihood(const KeypointMeasurement& msmt) const
// {
// 	// ROS_INFO_STREAM("Msmt class = " << msmt.class_measurements[msmt_id] << ", kp class = " << kp->classid);

// 	if (msmt.kp_class_id != classid_) return 0; //TODO
//     if (is_bad_) return 0;

//     bool include_S = true;

//     // gtsam::Pose3 G_x_I = state_estimate.at<gtsam::Pose3>(sym::X(msmt.pose_id));
//     gtsam::Pose3 G_x_I = graph_->calculateEstimate<gtsam::Pose3>(sym::X(msmt.pose_id));
//     gtsam::Pose3 G_x_C = G_x_I.compose(I_T_C_);
//     Eigen::Vector3d G_l;
//     if (in_graph_){
//         G_l = graph_->calculateEstimate<gtsam::Point3>(sym::L(global_id_)).vector();
//     } else {
//         G_l = global_position_;
//     }

//     // ROS_INFO_STREAM("KP global position = " << G_l.transpose());
//     // ROS_INFO_STREAM("Camera position = " << G_x_C.translation().transpose());

//     // make gtsam camera & project landmark's estimated position (note: px_msmt must be normalized!)
//     gtsam::CalibratedCamera camera(G_x_C);
//     gtsam::Matrix Hpose, Hpoint;
//     gtsam::Point2 zhat_pt;
//     try {
//         zhat_pt = camera.project(gtsam::Point3(G_l), Hpose, Hpoint);
//     } catch (gtsam::CheiralityException& e) {
//         return 0.0;
//     }
//     Eigen::Vector2d zhat = zhat_pt.vector();

//     Eigen::Matrix2d S_minus_R = Eigen::Matrix2d::Zero();

//     if (include_S) {
//         // full measurement function jacobian
//         Eigen::Matrix<double, 2, 9> H;
//         H.block<2,3>(0,0) = Hpoint;
//         H.block<2,6>(0,3) = Hpose;

//         // joint covariance matrix [Pll Plx ; Pxl Pxx]
//         Eigen::MatrixXd Plx = Eigen::MatrixXd::Zero(9,9);
//         if (in_graph_) {
//             JointMarginal joint_marginals = jointMarginalCovariance(msmt.measured_symbol);

//             Plx.block<3,3>(0,0) = joint_marginals(sym::L(global_id_), sym::L(global_id_));
//             Plx.block<6,6>(3,3) = joint_marginals(msmt.measured_symbol, msmt.measured_symbol);
//             Plx.block<3,6>(0,3) = joint_marginals(sym::L(global_id_), msmt.measured_symbol);
//             Plx.block<6,3>(3,0) = Plx.block<3,6>(0,3).transpose();
//         } else {
//             Plx.block<3,3>(0,0) = global_covariance_;
//             Plx.block<6,6>(3,3) = graph_->marginalCovariance(msmt.measured_symbol);
//         }

//         S_minus_R = H * Plx * H.transpose();
//     }

//     // msmt noise
//     // px_noise in terms of pixels but here we're in normalized coordinates so divide by focal length
//     Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
//     double px_sigma = msmt.pixel_sigma;
//     R(0,0) = px_sigma * px_sigma / (camera_calibration_->fx() * camera_calibration_->fx());
//     R(1,1) = px_sigma * px_sigma / (camera_calibration_->fy() * camera_calibration_->fy());

//     // p(z | x. l) ~ N(zhat; S)

//     Eigen::Vector2d residual = msmt.normalized_measurement - zhat;

//     double likelihood = normalPdf(residual, Eigen::Vector2d::Zero(), S_minus_R + R);

//     // double likelihood = std::exp( -0.5 * residual.transpose() * (S_minus_R + R).inverse() * residual );
//     // likelihood /= std::sqrt( (2 * M_PI * (S_minus_R + R)).determinant() );

//     // std::cout << "Received d = " << msmt.depth << ", expected d = " << range_hat << std::endl;
//     // std::cout << "ratio = " << range_hat / msmt.depth << std::endl;

//     // ROS_INFO_STREAM("Likelihood computation.");
//     // std::cout << "Keypoint ID " << global_id_ << std::endl;
//     // std::cout << "Object ID " << id_ << std::endl;
//     // std::cout << "residual = " << residual.transpose() << "\n";
//     // // std::cout << "Expected [" << zhat.transpose() << ", " << range_hat << "], got [" <<
//     msmt.normalized_measurement.transpose() << ", " << msmt.depth << "]" << std::endl;
//     // std::cout << "Msmt covariance: " << "\n" << S_minus_R + R << "\n";
//     // // std::cout << "S - R = \n" << S_minus_R << "\n";
//     // // std::cout << "det(S) = " << (S_minus_R + R).determinant() << "\n";
//     // std::cout << "Likelihood = " << likelihood << "\n";

//     return likelihood;
// }
