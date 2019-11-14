
#include "semantic_slam/ObjectHandler.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/MLDataAssociator.h"

#include <ros/package.h>

#include <boost/filesystem.hpp>

#include <visualization_msgs/MarkerArray.h>

using namespace std::string_literals;
namespace sym = symbol_shorthand;

void ObjectHandler::setup()
{
    ROS_INFO("Starting object handler.");
    std::string object_topic;
    pnh_.param("keypoint_detection_topic", object_topic, "/semslam/img_keypoints"s);

    ROS_INFO_STREAM("[ObjectHandler] Subscribing to topic " << object_topic);

    subscriber_ = nh_.subscribe(object_topic, 1000, &ObjectHandler::msgCallback, this);

    received_msgs_ = 0;
    measurements_processed_ = 0;
    n_landmarks_ = 0;

    node_chr_ = 'o';


    std::string base_path = ros::package::getPath("semantic_slam");
    std::string model_dir = base_path + std::string("/models/objects/");

    loadModelFiles(model_dir);
    loadCalibration();
    loadParameters();


    vis_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("keypoint_objects/object_markers", 10);
}


void ObjectHandler::loadModelFiles(std::string path) {
    boost::filesystem::path model_path(path);
    for (auto i = boost::filesystem::directory_iterator(model_path); i != boost::filesystem::directory_iterator(); i++) {
        if (!boost::filesystem::is_directory(i->path())) {
            ROS_INFO("Loading %s", i->path().filename().string().c_str());
            std::string file_name = i->path().filename().string();
            std::string class_name = file_name.substr(0, file_name.find('.'));

            object_models_[class_name] = geometry::readModelFile(i->path().string());
        }
        else {
            continue;
        }
    }
    ROS_INFO("Loaded %d models", (int)(object_models_.size()));
}

bool ObjectHandler::loadCalibration() {
    // Read from the ros parameter server...

    double fx, fy, s, u0, v0, k1, k2, p1, p2;
    s = 0.0; //skew
    if (!pnh_.getParam("cam_cx", u0) || 
        !pnh_.getParam("cam_cy", v0) || 
        !pnh_.getParam("cam_d0", k1) ||
        !pnh_.getParam("cam_d1", k2) ||
        !pnh_.getParam("cam_d2", p1) ||
        !pnh_.getParam("cam_d3", p2) || 
        !pnh_.getParam("cam_fx", fx) ||
        !pnh_.getParam("cam_fy", fy)) {
        ROS_ERROR("Error: unable to read camera intrinsic parameters.");
        return false;
    }

    camera_calibration_ = boost::make_shared<CameraCalibration>(fx, fy, s, u0, v0, k1, k2, p1, p2);

    std::vector<double> I_p_C, I_q_C;

    if (!getRosParam(pnh_, "I_p_C", I_p_C) ||
        !getRosParam(pnh_, "I_q_C", I_q_C)) {
        ROS_ERROR("Error: unable to read camera extrinsic parameters.");
        return false;
    }

    Eigen::Quaterniond q;
    Eigen::Vector3d p;

    q.x() = I_q_C[0];
    q.y() = I_q_C[1];
    q.z() = I_q_C[2];
    q.w() = I_q_C[3];
    q.normalize();

    p << I_p_C[0], I_p_C[1], I_p_C[2];

    I_T_C_ = Pose3(q,p);

    return true;
}

bool ObjectHandler::loadParameters()
{
    if (!pnh_.getParam("keypoint_activation_threshold", params_.keypoint_activation_threshold) ||

        !pnh_.getParam("min_object_n_keypoints", params_.min_object_n_keypoints) ||
        !pnh_.getParam("min_landmark_observations", params_.min_landmark_observations) ||

        !pnh_.getParam("structure_regularization_factor", params_.structure_regularization_factor) ||
        !pnh_.getParam("robust_estimator_parameter", params_.robust_estimator_parameter) ||

        !pnh_.getParam("structure_error_coefficient", params_.structure_error_coefficient) ||
        !pnh_.getParam("include_objects_in_graph", params_.include_objects_in_graph) ||

        !pnh_.getParam("new_landmark_weight_threshold", params_.new_landmark_weight_threshold) ||
        !pnh_.getParam("mahal_thresh_assign", params_.mahal_thresh_assign) ||
        !pnh_.getParam("mahal_thresh_init", params_.mahal_thresh_init) ||
        !pnh_.getParam("keypoint_initialization_depth_sigma", params_.keypoint_initialization_depth_sigma) ||
        !pnh_.getParam("constraint_weight_threshold", params_.constraint_weight_threshold)) {

        ROS_ERROR("Unable to load object handler parameters");
        return false;
    }

    return true;
}

bool ObjectHandler::keepFrame(ros::Time time)
{
    if (keyframes_.empty()) return true;

    auto last_keyframe = keyframes_.back();

    Pose3 relpose;
    bool got_relpose = odometry_handler_->getRelativePoseEstimate(last_keyframe->time(), time, relpose);

    if (!got_relpose) {
        ROS_WARN_STREAM("Too few odometry messages received to get keyframe relative pose");
        return true;
    }

    double translation_threshold = 0.05;
    double rotation_threshold = 10; // degrees

    if (relpose.translation().norm() > translation_threshold ||
        2*std::acos(relpose.rotation().w()) > rotation_threshold) {
        return true;
    } else {
        return false;
    }
}

void ObjectHandler::update()
{
    // Iterate over each of our pending measurement messages, try to add them to the 
    // factor graph

    // Start by collecting those messages that we have "spine" nodes for already in the graph
    // These vectors are in a 1-to-1 correspondence

    std::vector<object_pose_interface_msgs::KeypointDetections> messages;
    std::vector<CeresNodePtr> spine_nodes;
    bool got_msg = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // try only processing one message per call
        while (!msg_queue_.empty() && !got_msg) {
            auto msg = msg_queue_.front();

            if (keepFrame(msg.header.stamp)) {
                auto spine_node = odometry_handler_->getSpineNode(msg.header.stamp);
                if (spine_node) {
                    messages.push_back(msg);
                    spine_nodes.push_back(spine_node);
                    got_msg = true;
                }
            }

            msg_queue_.pop_front();
        }


        // Assume that the messages were received in chronological order, so we can stop after the first
        // we reach that we can't attach
        // while (!msg_queue_.empty()) {
        //     auto msg = msg_queue_.front();

        //     auto spine_node = odometry_handler_->getSpineNode(msg.header.stamp);

        //     if (spine_node) {
        //         messages.push_back(msg);
        //         spine_nodes.push_back(spine_node);
        //         msg_queue_.pop_front();
        //     } else {
        //         break;
        //     }
        // }
    }

    if (!got_msg) return;

    for (size_t i = 0; i < messages.size(); ++i) {
        auto& msg = messages[i];
        auto spine_node = boost::dynamic_pointer_cast<SE3Node>(spine_nodes[i]);

        // Build the ObjectMeasurement structures for all detected objects here
        auto keyframe = util::allocate_aligned<SemanticKeyframe>(msg.header.stamp, spine_node->key());
        keyframes_.push_back(keyframe);
        // aligned_vector<ObjectMeasurement> measurements;

        for (auto detection : msg.detections) {
            auto model_it = object_models_.find(detection.obj_name);
            if (model_it == object_models_.end()) {
                ROS_WARN_STREAM("No object model for class " << detection.obj_name);
            }

            geometry::ObjectModelBasis model = model_it->second;


            // Build vectors etc to optimize structure
            Eigen::MatrixXd normalized_coords(3, detection.x.size());
            Eigen::VectorXd weights(detection.x.size());

            for (size_t i = 0; i < detection.x.size(); ++i)
            {
                Eigen::Vector2d img_coords(detection.x[i], detection.y[i]);
                normalized_coords.block<2,1>(0,i) = camera_calibration_->calibrate(img_coords);
                normalized_coords(2, i) = 1.0;

                weights(i) = detection.probabilities[i];
            }

            geometry::StructureResult result;


            try
            {
                result = geometry::optimizeStructureFromProjection(normalized_coords,
                                                                    model,
                                                                    weights,
                                                                    true /* compute depth covariance */);

                // ROS_WARN_STREAM("Optimized structure.");
                // std::cout << "t = " << result.t.transpose() << std::endl;
            }
            catch (std::exception &e)
            {
                ROS_WARN_STREAM("Structure optimization failed:\n"
                                << e.what());
                continue;
            }


            // Build object measurement structure
            ObjectMeasurement obj_msmt;
            obj_msmt.observed_key = spine_node->key();
            obj_msmt.stamp = detection.header.stamp;
            obj_msmt.platform = "zed";
            obj_msmt.frame = detection.header.frame_id;
            obj_msmt.global_msmt_id = measurements_processed_++;
            obj_msmt.obj_name = detection.obj_name;
            obj_msmt.track_id = detection.bounding_box.id;
            obj_msmt.t = result.t;
            obj_msmt.q = Eigen::Quaterniond(result.R);

            obj_msmt.bbox.xmin = detection.bounding_box.xmin;
            obj_msmt.bbox.ymin = detection.bounding_box.ymin;
            obj_msmt.bbox.xmax = detection.bounding_box.xmax;
            obj_msmt.bbox.ymax = detection.bounding_box.ymax;

            double bbox_dim = 0.5 * (obj_msmt.bbox.xmax - obj_msmt.bbox.xmin +
                                     obj_msmt.bbox.ymax - obj_msmt.bbox.ymin);

            size_t n_keypoints_observed = 0;

            for (size_t i = 0; i < detection.x.size(); ++i) {
                KeypointMeasurement kp_msmt;

                kp_msmt.measured_key = spine_node->key();
                kp_msmt.stamp = detection.header.stamp;
                kp_msmt.platform = "zed";
                kp_msmt.obj_name = detection.obj_name;

                kp_msmt.pixel_measurement << detection.x[i], detection.y[i];
                kp_msmt.normalized_measurement = normalized_coords.block<2,1>(0, i);

                kp_msmt.score = detection.probabilities[i];
                kp_msmt.depth = result.Z(i);

                // covariance estimation in the structure optimization may have failed
                if (result.Z_covariance.size() > 0) {
                    kp_msmt.depth_sigma = std::sqrt(result.Z_covariance(i));
                } else {
                    // TODO what here?
                    kp_msmt.depth_sigma = 5;
                }

                kp_msmt.kp_class_id = i;

                // TODO what here?
                kp_msmt.pixel_sigma = 0.1 * bbox_dim;

                if (kp_msmt.score > params_.keypoint_activation_threshold)
                {
                    kp_msmt.observed = true;
                    n_keypoints_observed++;
                } else {
                    kp_msmt.observed = false;
                }

                obj_msmt.keypoint_measurements.push_back(kp_msmt);
            }

            if (n_keypoints_observed > 0) {
                keyframe->measurements.push_back(obj_msmt);
            }
        }

        if (keyframe->measurements.size() > 0) {

            // Create the list of measurements we need to associate.
            // Identify which measurements have been tracked from already known objects
            std::vector<size_t> measurement_index;
            std::map<size_t, size_t> known_das;

            for (size_t i = 0; i < keyframe->measurements.size(); ++i) 
            {
                auto known_id_it = object_track_ids_.find(keyframe->measurements[i].track_id);
                if (known_id_it == object_track_ids_.end()) {
                    // track not known, perform actual data association
                    measurement_index.push_back(i);
                } else {
                    // track ID known, take data association as given
                    known_das[i] = known_id_it->second;
                }
            }

            // Create the list of objects we need to associate.
            // count visible landmarks & create mapping from list of visible to list of all
            std::vector<bool> visible = getVisibleObjects(spine_node);
            size_t n_visible = 0;
            std::vector<size_t> object_index;
            for (size_t j = 0; j < estimated_objects_.size(); ++j)
            {
                if (visible[j])
                {
                    n_visible++;
                    estimated_objects_[j]->setIsVisible(spine_node);
                    object_index.push_back(j);
                }
            }

            Eigen::MatrixXd mahals = Eigen::MatrixXd::Zero(measurement_index.size(), n_visible);
            for (size_t i = 0; i < measurement_index.size(); ++i)
            {
                // ROS_WARN_STREAM("** Measurement " << i << "**");
                for (size_t j = 0; j < n_visible; ++j)
                {
                    mahals(i, j) = estimated_objects_[object_index[j]]->computeMahalanobisDistance(
                                                                            keyframe->measurements[measurement_index[i]]);
                }
            }

            // std::cout << "Mahals:\n" << mahals << std::endl;
      
            Eigen::MatrixXd weights_matrix = MLDataAssociator(params_).computeConstraintWeights(mahals);

            updateObjects(spine_node, 
                          keyframe->measurements, 
                          measurement_index, 
                          known_das, 
                          weights_matrix, 
                          object_index);
        }
    }

    visualizeObjectMeshes();
}

bool ObjectHandler::updateObjects(SE3Node::Ptr node,
                                  const aligned_vector<ObjectMeasurement>& measurements,
                                  const std::vector<size_t>& measurement_index,
                                  const std::map<size_t, size_t>& known_das,
                                  const Eigen::MatrixXd& weights,
                                  const std::vector<size_t>& object_index)
{
    if (measurements.size() == 0) return true;

    Pose3 map_T_body = node->pose();
    Pose3 map_T_camera = map_T_body * I_T_C_;

    /** new objects **/

    for (size_t k = 0; k < measurement_index.size(); ++k)
    {
        if (weights(k, weights.cols() - 1) >= params_.new_landmark_weight_threshold)
        {
            auto& msmt = measurements[measurement_index[k]];

            EstimatedObject::Ptr new_obj =
                EstimatedObject::create(graph_, params_, object_models_[msmt.obj_name], estimated_objects_.size(),
                                        n_landmarks_, msmt, 
                                        map_T_camera, /* G_T_C */
                                        I_T_C_,       
                                        "zed", camera_calibration_);

            new_obj->addKeypointMeasurements(msmt, weights(k, weights.cols() - 1));
            n_landmarks_ += new_obj->numKeypoints();

            estimated_objects_.push_back(new_obj);

            ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: initializing new object {}, weight {}.",
                                        k,
                                        estimated_objects_.size() - 1,
                                        msmt.obj_name,
                                        weights(k, weights.cols() - 1)));

            // ROS_INFO_STREAM("Measurement " << k << ": initializing new object " << estimated_objects_.size() - 1 << ", class "
            //                                 << msmt.obj_name << ", weight " << weights(k, weights.cols() - 1));

            object_track_ids_[msmt.track_id] = estimated_objects_.size() - 1;
        }
    }
    /** existing objects **/

    // existing objects that were tracked
    for (const auto& known_da : known_das) {
        auto& msmt = measurements[known_da.first];

        ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: adding factors from {} to object {} [{}] (tracked).",
                                    known_da.first,
                                    msmt.obj_name,
                                    DefaultKeyFormatter(msmt.observed_key),
                                    known_da.second,
                                    estimated_objects_[known_da.second]->obj_name()));

        // ROS_INFO_STREAM("Measurement " << known_da.first << ": adding factors from "
        //                             << DefaultKeyFormatter(msmt.observed_key) << " to object "
        //                             << known_da.second << " with weight 1 [tracked]");

        estimated_objects_[known_da.second]->addKeypointMeasurements(msmt, 1.0);

    }

    // existing objects that were associated
    for (size_t k = 0; k < measurement_index.size(); ++k)
    {
        for (int j = 0; j < weights.cols() - 1; ++j)
        {
            if (weights(k, j) >= params_.constraint_weight_threshold)
            {
                auto& msmt = measurements[measurement_index[k]];

                ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: adding factors from {} "
                                                "to object {} [{}] with weight {}.",
                                            measurement_index[k],
                                            msmt.obj_name,
                                            DefaultKeyFormatter(msmt.observed_key),
                                            object_index[j],
                                            estimated_objects_[object_index[j]]->obj_name(),
                                            weights(k, j)));

                estimated_objects_[object_index[j]]->addKeypointMeasurements(msmt, weights(k, j));

                // update information about track ids
                object_track_ids_[msmt.track_id] = object_index[j];
            }
        }
    }

    return true;

}

std::vector<bool> 
ObjectHandler::getVisibleObjects(SE3Node::Ptr node)
{
    std::vector<bool> visible(estimated_objects_.size(), false);

    Pose3 map_T_body = node->pose();

    // Pose3 G_T_C = map_T_body * I_T_C_;

    // Do simple range check
    for (size_t i = 0; i < estimated_objects_.size(); ++i)
    {
        if (estimated_objects_[i]->bad())
            continue;

        // Pose3 map_T_obj = graph_->getNode<SE3Node>(sym::O(estimated_objects_[i]->id()))->pose();
        Pose3 map_T_obj = estimated_objects_[i]->pose();

        Pose3 body_T_obj = map_T_body.inverse() * map_T_obj;

        if (body_T_obj.translation().norm() <= 75)
        {
            visible[i] = true;
        }
    }

    return visible; 
}

void ObjectHandler::msgCallback(const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg)
{
    // ROS_INFO_STREAM("[ObjectHandler] Received keypoint detection message, t = " << msg->header.stamp);
	received_msgs_++;

	if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM("[ObjectHandler] Error: dropped keypoint message. Expected " << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }
    // ROS_INFO_STREAM("Received relpose msg, seq " << msg->header.seq << ", time " << msg->header.stamp);
    {
    	std::lock_guard<std::mutex> lock(mutex_);
    	msg_queue_.push_back(*msg);
    }
    // cv_.notify_all();

    last_msg_seq_ = msg->header.seq;
}

void ObjectHandler::visualizeObjectMeshes() const
{
    // Clear screen first
    // visualization_msgs::MarkerArray reset_markers;
    // reset_markers.markers.resize(1);
    // reset_markers.markers[0].header.frame_id = "map";
    // reset_markers.markers[0].header.stamp = ros::Time::now();
    // reset_markers.markers[0].ns = "deleteAllMarkers";
    // reset_markers.markers[0].action = 3;
    // vis_pub_.publish(reset_markers);    

    visualization_msgs::MarkerArray object_markers;
    visualization_msgs::Marker object_marker;
    object_marker.type = visualization_msgs::Marker::MESH_RESOURCE;

    object_marker.header.frame_id = "map";
    object_marker.header.stamp = ros::Time::now();

    double model_scale = 1;

    object_marker.scale.x = model_scale;
    object_marker.scale.y = model_scale;
    object_marker.scale.z = model_scale;

    object_marker.color.r = 0;
    object_marker.color.g = 0;
    object_marker.color.b = 1;
    object_marker.color.a = 1.0f;

    object_marker.ns = "objects";
    object_marker.action = visualization_msgs::Marker::ADD;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "objects";
    delete_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    delete_marker.action = visualization_msgs::Marker::DELETE;

    size_t n_added = 0;

    for (const EstimatedObject::Ptr& obj : estimated_objects_) {
        if (obj->bad()) {
            delete_marker.id = obj->id();
            object_markers.markers.push_back(delete_marker);
            continue;
        }

        // if (!obj->inGraph()) continue;

        if (obj->inGraph()) {
            // blue
            object_marker.color.b = 1.0;
            object_marker.color.r = 0.0;
            object_marker.color.a = 1.0;
        } else {
            // red
            object_marker.color.b = 0.0;
            object_marker.color.r = 1.0;
            object_marker.color.a = 0.5;
        }


        object_marker.action = visualization_msgs::Marker::ADD;
        // object_text.action = visualization_msgs::Marker::ADD;

        double model_scale = 1.0;

        if (obj->obj_name() == "chair") {
            model_scale = 2;
        } else if (obj->obj_name() == "gascan") {
            model_scale = 0.25;
        } else if (obj->obj_name() == "cart") {
            model_scale = 0.12;
        } else if (obj->obj_name() == "tableclosed") {
            model_scale = 10.0;
        } else if (obj->obj_name() == "ladder") {
            model_scale = 0.125;
        } else if (obj->obj_name() == "pelican") {
            model_scale = 0.25;
        } 

        object_marker.scale.x = model_scale;
        object_marker.scale.y = model_scale;
        object_marker.scale.z = model_scale;

        object_marker.pose.position.x = obj->pose().x();
        object_marker.pose.position.y = obj->pose().y();
        object_marker.pose.position.z = obj->pose().z();

        Eigen::Quaterniond q = obj->pose().rotation();

        object_marker.pose.orientation.x = q.x();
        object_marker.pose.orientation.y = q.y();
        object_marker.pose.orientation.z = q.z();
        object_marker.pose.orientation.w = q.w();

        object_marker.mesh_resource = std::string("package://semantic_slam/models/viz_meshes/") + obj->obj_name() + ".dae";
        // object_marker.mesh_resource = std::string("package://semslam/models/viz_meshes/car.dae");

        // object_text.pose.position.x = obj->pose().translation().x();
        // object_text.pose.position.y = obj->pose().translation().y();
        // object_text.pose.position.z = Z_SCALE * obj->pose().translation().z() + 1.5;

        object_marker.id = obj->id();
        object_marker.text = obj->obj_name();

        object_markers.markers.push_back(object_marker);


        std::vector<int64_t> keypoint_ids = obj->getKeypointIndices();
        // addKeypointMarkers(object_markers.markers, model_scale, obj, t);

        ++n_added;
    }

    vis_pub_.publish(object_markers);
}

void ObjectHandler::visualizeObjects() const
{
  // Clear screen first
  visualization_msgs::MarkerArray reset_markers;
  reset_markers.markers.resize(1);
  reset_markers.markers[0].header.frame_id = "map";
  reset_markers.markers[0].header.stamp = ros::Time::now();
  reset_markers.markers[0].ns = "deleteAllMarkers";
  reset_markers.markers[0].action = 3;
  vis_pub_.publish(reset_markers);

  visualization_msgs::MarkerArray markers_message;

  double scale = 2;

  // ** Full object markers **

  visualization_msgs::Marker marker;

  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;

  marker.color.r = 0;
  marker.color.g = 0;
  marker.color.b = 1;
  marker.color.a = .8;

  marker.scale.x = scale / 8.0;
  marker.scale.y = scale / 8.0;
  marker.scale.z = scale / 8.0;

  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();

  marker.ns = "objects";

  visualization_msgs::Marker object_text;

  object_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  object_text.header.frame_id = "map";
  object_text.header.stamp = ros::Time::now();
  object_text.action = visualization_msgs::Marker::ADD;
  object_text.ns = "object_names";

  object_text.scale.z = 1;

  object_text.color.r = 0;
  object_text.color.g = 0;
  object_text.color.b = 0;
  object_text.color.a = 1.0f;

  for (size_t i = 0; i < estimated_objects_.size(); ++i)
  {
    if (estimated_objects_[i]->bad())
      continue;

    if (estimated_objects_[i]->inGraph())
    {
      marker.color.r = 0;
      marker.color.g = 1;
      marker.color.b = 0;
      marker.color.a = 1.0;
    }
    else
    {
      marker.color.r = 1;
      marker.color.g = 0;
      marker.color.b = 0;
      marker.color.a = 0.5f;
    }

    marker.id = i;

    Pose3 pose = estimated_objects_[i]->pose();
    marker.pose.position.x = pose.x();
    marker.pose.position.y = pose.y();
    marker.pose.position.z = pose.z();

    markers_message.markers.push_back(marker);

    object_text.id = i;
    object_text.pose = marker.pose;
    object_text.pose.position.z += 1;
    object_text.text = fmt::format("{}-{}", estimated_objects_[i]->obj_name(), i);

    markers_message.markers.push_back(object_text);
  }

//   // ** Component keypoint markers **
//   scale = 1.0;

//   marker.color.g = 1;
//   marker.color.b = 0;
//   marker.color.a = 1;

//   marker.scale.x = scale / 8.0;
//   marker.scale.y = scale / 8.0;
//   marker.scale.z = scale / 8.0;

//   marker.ns = "keypoints";

//   object_text.ns = "keypoint_ids";
//   object_text.scale.z = 0.5;

//   for (auto &o : estimated_objects_)
//   {
//     // if (o->bad() || !o->inGraph()) continue;
//     const auto &kps = o->getKeypoints();

//     for (size_t i = 0; i < kps.size(); ++i)
//     {
//       if (kps[i]->bad())
//       {
//         marker.action = visualization_msgs::Marker::DELETE;
//       }
//       else
//       {
//         marker.action = visualization_msgs::Marker::ADD;
//       }

//       if (kps[i]->inGraph())
//       {
//         marker.color.g = 1;
//         marker.color.r = 0;
//       }
//       else
//       {
//         marker.color.g = 0;
//         marker.color.r = 1;
//       }

//       marker.id = kps[i]->id();

//       marker.pose.position.x = kps[i]->position().x();
//       marker.pose.position.y = kps[i]->position().y();
//       marker.pose.position.z = kps[i]->position().z();

//       markers_message.markers.push_back(marker);

//       object_text.pose.position = marker.pose.position;
//       object_text.pose.position.z += 0.5;
//       object_text.id = kps[i]->id();
//       // object_text.text = fmt::format("{}-{}", o->id(), kps[i]->classid());
//       object_text.text = fmt::format("{}", kps[i]->classid());
//       markers_message.markers.push_back(object_text);
//     }
//   }

  vis_pub_.publish(markers_message);
}