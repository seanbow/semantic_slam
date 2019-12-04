#pragma once

#include <ros/ros.h>

#include <vector>
#include <unordered_map>

#include "semantic_slam/Utils.h"

/// Integer nonlinear key type
typedef std::uint64_t Key;

template <
    class Key,
    class T,
    class Hash = std::hash<Key>,
    class KeyEqual = std::equal_to<Key>,
    class Allocator = Eigen::aligned_allocator<std::pair<const Key, T>>
> 
using aligned_unordered_map = std::unordered_map<Key,T,Hash,KeyEqual,Allocator>;

template <
    class Key,
    class T,
    class Hash = std::hash<Key>,
    class KeyEqual = std::equal_to<Key>,
    class Allocator = Eigen::aligned_allocator<std::pair<const Key, T>>
> 
using aligned_map = std::unordered_map<Key,T,Hash,KeyEqual,Allocator>;

template <class T, class Allocator = Eigen::aligned_allocator<T>>
using aligned_vector = std::vector<T, Allocator>;

enum class OptimizationBackend {
  CERES,
  GTSAM
};

struct ObjectParams
{
  double max_new_factor_error;
  double keypoint_initialization_depth_sigma;
  int min_landmark_observations;  //< minimum number of observations before triangulating & including in optim.
  double camera_range;				 //< maximum range at which camera can detect objects

  double calibration_tolerance;  //< when undistorting a point, enforce that its reprojection error is < this

  double mahal_thresh_assign;  //< Mahalanobis distance threshold below which to assign data
  double mahal_thresh_init;	//< Mahal. distance threshold above which to initialize new mapped landmarks

  double keypoint_activation_threshold;  //< threshold on heatmap above which to consider a keypoint observation valid
  double keypoint_msmt_sigma;			 //< keypoint measurement sigma for observed keypoints

  double structure_regularization_factor;

  OptimizationBackend optimization_backend; // CERES or GTSAM

  double landmark_merge_threshold;		 //< Threshold on between-landmark bhattacharyya distance below which to merge landmarks
  double probability_landmark_new;		 //< fixed probability that a measurement corresponds to a new landmark
  double constraint_weight_threshold;	//< if weight of a constraint is < this do not add to factor graph
  double new_landmark_weight_threshold;  //< same as above but for new landmark initializations

  int min_object_n_keypoints;  //< minimum number of observed/estimated keypoints to consider an object good

  int min_observed_keypoints_to_initialize;

  double keyframe_translation_threshold;
  double keyframe_rotation_threshold;

  char object_symbol_char;
  char keypoint_symbol_char;
  char spur_node_symbol_char;

  bool include_objects_in_graph;
  bool include_depth_constraints;
  double point_cloud_depth_sigma;

  double structure_error_coefficient;

  double robust_estimator_parameter;

  ObjectParams() : calibration_tolerance(1e-2)
  {
  }
};

struct BoundingBox
{
    double xmin;
    double ymin;
    double xmax;
    double ymax;
};

struct KeypointMeasurement
{
  // uint64_t pose_id;
  Key measured_key;
  ros::Time stamp;
  std::string platform;

  Eigen::Vector2d normalized_measurement;
  Eigen::Vector2d pixel_measurement;
  bool observed;
  double pixel_sigma;
  size_t kp_class_id;
  double score;
  // size_t object_class_id;
  std::string obj_name;

  Eigen::Vector3d structure;
  double depth;
  double depth_sigma;

  double measured_depth;
  double measured_depth_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Data from the set of keypoint measurements received from a single object detection
 */
struct ObjectMeasurement
{
  Key observed_key;
  // uint64_t pose_id;
  uint64_t global_msmt_id;
  ros::Time stamp;
  std::string platform;
  std::string frame;

  // size_t class_id;
  std::string obj_name;

  size_t track_id; //< If this object was tracked along multiple images, the id of the track

  BoundingBox bbox;

  aligned_vector<KeypointMeasurement> keypoint_measurements;

  Eigen::Vector3d t;
  Eigen::Quaterniond q;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};